import argparse
from pathlib import Path
from dataclasses import asdict
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import MyGPT
from src.data import KoAlpacaDataset, collate_fn
from src.config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyGPT")
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        required=True,
    )
    parser.add_argument(
        "--weights-dir",
        "--weights_dir",
        dest="weights_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--wandb",
        type=str,
        required=False,
        default="MyGPT"
    )
    return parser.parse_args()

def generate_sample(model, tokenizer, device, instruction="인공지능의 미래는 어떻게 될까요?", max_new_tokens=50):
    model.eval()  

    prompt_text = f"[사용자]\n{instruction}\n\n[챗봇]\n"

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    generated_ids = input_ids

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text

def main():
    args = parse_args()
    config = TrainConfig.load_from_yaml(args.config_path)
    
    run = wandb.init(
        project=args.wandb,
        config=asdict(config)
    )
    
    run_weights_dir = args.weights_dir / run.name
    run_weights_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(**asdict(config.tokenizer))

    train_dataset = KoAlpacaDataset(dataset, tokenizer, config.model.max_seq_len)
    train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = MyGPT(n_embeddings=train_dataset.tokenizer.vocab_size, **asdict(config.model))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        epoch_train_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * input_ids.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        sample_text = generate_sample(model, tokenizer, device)
        print(f"\n[Epoch {epoch+1}]\nLoss: {epoch_train_loss}\n{sample_text}\n")

        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": epoch_train_loss,
            "generated_sample": wandb.Html(f"<pre>{sample_text}</pre>")
        })
        
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            },
            run_weights_dir / f"checkpoint-epoch-{epoch + 1}.pt",
        )
if __name__ == "__main__":
    main()