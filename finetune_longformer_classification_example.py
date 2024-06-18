import torch
from transformers import AutoTokenizer, LongformerForSequenceClassification



def test():

    tokenizer = AutoTokenizer.from_pretrained("jpwahle/longformer-base-plagiarism-detection")
    model = LongformerForSequenceClassification.from_pretrained("jpwahle/longformer-base-plagiarism-detection",
                                                                problem_type="multi_label_classification")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

    # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    num_labels = len(model.config.id2label)
    model = LongformerForSequenceClassification.from_pretrained(
        "jpwahle/longformer-base-plagiarism-detection", num_labels=num_labels, problem_type="multi_label_classification"
    )

    labels = torch.sum(
        torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    ).to(torch.float)
    loss = model(**inputs, labels=labels).loss
    print(loss)








if __name__ == "__main__":
    test()