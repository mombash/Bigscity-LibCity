#!/usr/bin/env python3
"""
RegMean Model Merging Script
Converted from Jupyter notebook to Python script

This script demonstrates model merging using RegMean algorithm on GLUE tasks.
It trains individual models on CoLA and SST-2 tasks, then merges them using
both RegMean and simple averaging methods.
"""

import os 
import re

import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding


def get_metrics_func(task_name):
    """Get metrics function for GLUE tasks."""
    metric = load_metric("glue", task_name)
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    return compute_metrics


def train_glue_model(task_name, checkpoint_name):
    """Train a model on a GLUE task."""
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base').cuda()
    ds = load_dataset('glue', task_name)
    metric = get_metrics_func(task_name)
    enc_ds = ds.map(lambda examples: tokenizer(examples["sentence"], max_length=128, truncation=True), batched=True)
    training_args = TrainingArguments(
        output_dir=f'./results/{task_name}',  # output directory
        num_train_epochs=3,                   # total # of training epochs
        per_device_train_batch_size=16,       # batch size per device during training
        per_device_eval_batch_size=16,        # batch size for evaluation
        warmup_steps=500,                     # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                    # strength of weight decay
    )
    trainer = Trainer(
        model=model,                        # the instantiated 🤗 Transformers model to be trained
        args=training_args,                 # training arguments, defined above
        train_dataset=enc_ds['train'],      # training dataset
        eval_dataset=enc_ds['validation'],  # evaluation dataset
        compute_metrics=metric,
        tokenizer=tokenizer,
    )

    checkpoint_path = os.path.join(training_args.output_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).cuda()
    else:
        trainer.train()

    return trainer, model


def filter_params_to_merge(param_names, exclude_param_regex):
    """Filter parameters to merge based on regex patterns."""
    params_to_merge = []
    for name in param_names:
        valid = not any([re.match(patt, name) for patt in exclude_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge


def filter_modules_by_regex(base_module, include_patterns, include_type):
    """Filter modules by regex patterns and types."""
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules


def compute_gram(model, trainer):
    """Compute gram matrices for each linear layer inputs."""
    train_dataloader = trainer.get_train_dataloader()
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            x = input[0].detach() # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0,1), x) # [h,h]
            if name not in grams:
                grams[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    n_step = 1000
    total = n_step if n_step > 0 else len(train_dataloader)
    for step, inputs in tqdm(enumerate(train_dataloader), total=total, desc='Computing gram matrix'):
        if n_step > 0 and step == n_step:
            break

        inputs = trainer._prepare_inputs(inputs)
        outputs = model(**inputs)

    for handle in handles:
        handle.remove()

    return grams


def avg_merge(local_models, global_model, regmean_grams=None, **kwargs):
    """Merge models using either RegMean or simple averaging."""
    params = {}
    for local_model in local_models:
        n2p = {k: v for k,v in local_model.named_parameters()}
        merge_param_names = filter_params_to_merge([n for n in n2p], ['.*classifier.*']) # for glue label spaces are different
        for n in merge_param_names:
            if n not in params:
                params[n] = []
            params[n].append(n2p[n])

    if regmean_grams: # regmean average
        avg_params = regmean_merge(params, regmean_grams)
    else: # simple average
        avg_params = {k: torch.stack(v,0).mean(0) for k, v in params.items()}

    return avg_params


def copy_params_to_model(avg_params, model):
    """Copy averaged parameters to model."""
    for n, p in model.named_parameters():
        if n in avg_params:
            p.data.copy_(avg_params[n])


def reduce_non_diag(cov_mat, a):
    """Reduce non-diagonal elements of covariance matrix."""
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret


def regmean_merge(all_params, all_grams):
    """Merge parameters using RegMean algorithm."""
    avg_params = {}
    n_model = len(all_grams)
    for name in all_params:
        h_avged = False
        if name.endswith('.weight'):
            print(f'Regmean: {name}')
            module_name = name[:-len('.weight')]
            if module_name in all_grams[0]:
                gram_m_ws, grams = [], []

                for model_id, model_grams in enumerate(all_grams):
                    param_grams = model_grams[module_name]

                    # for roberta we dont need this; but it is important for deberta and t5
                    #param_grams = reduce_non_diag(param_grams, a=0.9)

                    param = all_params[name][model_id]
                    gram_m_ws.append(torch.matmul(param_grams, param.transpose(0,1)))
                    grams.append(param_grams)
                sum_gram = sum(grams)
                sum_gram_m_ws = sum(gram_m_ws)
                sum_gram_inv = torch.inverse(sum_gram)
                wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                w = wt.transpose(0,1)
                avg_params[name] = w
                h_avged = True
        if not h_avged: # if not averaged with regmean, then do simple avg
            avg_params[name] = torch.stack(all_params[name],0).mean(0)
           
    return avg_params


def main():
    """Main execution function."""
    print("RegMean Model Merging Script")
    print("=" * 50)
    
    # Print versions
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Train individual models
    print("\n" + "=" * 50)
    print("Training individual models")
    print("=" * 50)
    
    print("\nTraining CoLA model...")
    trainer1, model1 = train_glue_model('cola', 'checkpoint-1500')
    
    print("\nTraining SST-2 model...")
    trainer2, model2 = train_glue_model('sst2', 'checkpoint-12500')
    
    # Compute gram matrices
    print("\n" + "=" * 50)
    print("Computing gram matrices")
    print("=" * 50)
    
    print("\nComputing gram matrix for CoLA model...")
    with torch.no_grad():
        grams1 = compute_gram(model1, trainer1)
    
    print("\nComputing gram matrix for SST-2 model...")
    with torch.no_grad():
        grams2 = compute_gram(model2, trainer2)
    
    # Performance before merging
    print("\n" + "=" * 50)
    print("Performance before merging")
    print("=" * 50)
    
    print("\nCoLA model performance:")
    cola_results = trainer1.evaluate()
    print(f"  Matthews Correlation: {cola_results['eval_matthews_correlation']:.4f}")
    print(f"  Loss: {cola_results['eval_loss']:.4f}")
    
    print("\nSST-2 model performance:")
    sst_results = trainer2.evaluate()
    print(f"  Accuracy: {sst_results['eval_accuracy']:.4f}")
    print(f"  Loss: {sst_results['eval_loss']:.4f}")
    
    # Merging with RegMean
    print("\n" + "=" * 50)
    print("Merging with RegMean")
    print("=" * 50)
    
    print("\nCreating merged model...")
    merged_model = AutoModelForSequenceClassification.from_pretrained('roberta-base').cuda()
    
    print("\nPerforming RegMean merge...")
    regmean_avg_params = avg_merge([model1, model2], merged_model, regmean_grams=[grams1, grams2])
    
    # Test merged model on CoLA
    print("\nTesting merged model on CoLA...")
    copy_params_to_model(regmean_avg_params, merged_model)
    merged_model.classifier = model1.classifier  # we didn't merge classification heads
    
    evaluator_cola = Trainer(
        model=merged_model,
        args=trainer1.args,
        train_dataset=trainer1.train_dataset,
        eval_dataset=trainer1.eval_dataset,
        compute_metrics=get_metrics_func('cola'),
        tokenizer=tokenizer,
    )
    
    cola_merged_results = evaluator_cola.evaluate()
    print(f"  Matthews Correlation: {cola_merged_results['eval_matthews_correlation']:.4f}")
    print(f"  Loss: {cola_merged_results['eval_loss']:.4f}")
    
    # Test merged model on SST-2
    print("\nTesting merged model on SST-2...")
    copy_params_to_model(regmean_avg_params, merged_model)
    merged_model.classifier = model2.classifier  # we didn't merge classification heads
    
    evaluator_sst = Trainer(
        model=merged_model,
        args=trainer2.args,
        train_dataset=trainer2.train_dataset,
        eval_dataset=trainer2.eval_dataset,
        compute_metrics=get_metrics_func('sst2'),
        tokenizer=tokenizer,
    )
    
    sst_merged_results = evaluator_sst.evaluate()
    print(f"  Accuracy: {sst_merged_results['eval_accuracy']:.4f}")
    print(f"  Loss: {sst_merged_results['eval_loss']:.4f}")
    
    # Merging with Simple Average
    print("\n" + "=" * 50)
    print("Merging with Simple Average")
    print("=" * 50)
    
    # Test simple average on CoLA
    print("\nTesting simple average on CoLA...")
    simple_avg_params = avg_merge([model1, model2], merged_model)
    copy_params_to_model(simple_avg_params, merged_model)
    merged_model.classifier = model1.classifier
    
    cola_simple_results = evaluator_cola.evaluate()
    print(f"  Matthews Correlation: {cola_simple_results['eval_matthews_correlation']:.4f}")
    print(f"  Loss: {cola_simple_results['eval_loss']:.4f}")
    
    # Test simple average on SST-2
    print("\nTesting simple average on SST-2...")
    simple_avg_params = avg_merge([model1, model2], merged_model)
    copy_params_to_model(simple_avg_params, merged_model)
    merged_model.classifier = model2.classifier
    
    sst_simple_results = evaluator_sst.evaluate()
    print(f"  Accuracy: {sst_simple_results['eval_accuracy']:.4f}")
    print(f"  Loss: {sst_simple_results['eval_loss']:.4f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("\nOriginal Models:")
    print(f"  CoLA - Matthews Correlation: {cola_results['eval_matthews_correlation']:.4f}")
    print(f"  SST-2 - Accuracy: {sst_results['eval_accuracy']:.4f}")
    
    print("\nRegMean Merged Model:")
    print(f"  CoLA - Matthews Correlation: {cola_merged_results['eval_matthews_correlation']:.4f}")
    print(f"  SST-2 - Accuracy: {sst_merged_results['eval_accuracy']:.4f}")
    
    print("\nSimple Average Merged Model:")
    print(f"  CoLA - Matthews Correlation: {cola_simple_results['eval_matthews_correlation']:.4f}")
    print(f"  SST-2 - Accuracy: {sst_simple_results['eval_accuracy']:.4f}")
    
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main() 