from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sys

def load_model(model_path):
    print("正在加载模型(CPU模式)，请耐心等待...")
    start_time = time.time()
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        pad_token='<|endoftext|>'  # 使用Qwen的EOS token作为pad token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 内存优化配置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    ).eval()
    print(f"加载完成，耗时 {time.time() - start_time:.1f}秒")
    print("提示：输入内容后请耐心等待响应(CPU推理较慢)\n输入'退出'结束对话\n")
    return tokenizer, model

def generate_response(model, tokenizer, prompt):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        # 确保使用正确的pad token
        if tokenizer.pad_token_id is None:
            inputs.pop('attention_mask', None)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask if 'attention_mask' in inputs else None,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    except Exception as e:
        return f"[生成错误] {str(e)}"

def chat_loop(tokenizer, model):
    while True:
        try:
            user_input = input("用户: ").strip()
            if user_input.lower() == 'exit':
                print("对话结束")
                break
            if not user_input:
                continue
            print("AI: ", end="", flush=True)
            start_time = time.time()
            response = generate_response(model, tokenizer, user_input)
            for i in range(0, len(response), 80):
                print(response[i:i+80])
                time.sleep(0.1)
            print(f"\n[响应时间: {time.time() - start_time:.1f}秒]")
        except KeyboardInterrupt:
            print("\n提示: 输入'退出'结束程序")
        except Exception as e:
            print(f"系统错误: {str(e)}")

if __name__ == "__main__":
    torch.set_num_threads(4)
    model_path = "/mnt/workspace/Qwen-1_8B-Chat"
    try:
        tokenizer, model = load_model(model_path)
        chat_loop(tokenizer, model)
    except Exception as e:
        print(f"初始化错误: {str(e)}")
        sys.exit(1)