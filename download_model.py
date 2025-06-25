from transformers import AutoTokenizer,AutoModelForCausalLM
def download_model(model_name: str, save_directory: str):
    """
    下载指定的模型和分词器，并保存到本地目录。

    Args:
        model_name (str): 模型名称，例如 "gpt2" 或 "bert-base-uncased"。
        save_directory (str): 保存模型的本地目录路径。
    """
    # 下载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.save_pretrained(save_directory)

    # 下载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.save_pretrained(save_directory)

    # print(f"模型和分词器已保存到 {save_directory}")

if __name__ == "__main__":
    # 示例：下载 GPT-2 模型
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    save_directory = "./models/gpt2"
    download_model(model_name, save_directory)