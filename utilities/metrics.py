import numpy as np
import evaluate
from sklearn.metrics import classification_report

accuracy = evaluate.load("accuracy")

def compute_classification_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def compute_classification_eval_report(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return classification_report(labels, predictions, output_dict=True)

def compute_led_classification_eval_report(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=1)
    return classification_report(labels, predictions, output_dict=True)



def create_multitask_classification_eval_metric(attributes_info, flatdict=True):
    def compute_multitask_metrics(eval_pred):
        predictions, labels = eval_pred
        eval_report = {}
        task_index = 0
        for att, info in attributes_info.items():
            index_range = info["logits_range"]
            attribute_index = info["attribute_index"]
            task_labels = labels[:, attribute_index]
            task_logits = predictions[:, index_range[0]:index_range[-1]]
            task_predictions = np.argmax(task_logits, axis=1)
            eval_report[att] = classification_report(task_labels, task_predictions, output_dict=True)
            task_index += 1
        if flatdict:
            flat_eval_report = {}
            for k, report in eval_report.items():
                for m, v in report.items():
                    if isinstance(v, dict):
                        for n, score in v.items():
                            metric_key = "_".join([k, m, n])
                            flat_eval_report[metric_key] = score
                    else:
                        metric_key = "_".join([k, m])
                        flat_eval_report[metric_key] = v
            return flat_eval_report
        return eval_report

    return compute_multitask_metrics

def create_rogue_matric(tokenizer):
    # load rouge
    rouge = evaluate.load('rouge')
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    return compute_metrics