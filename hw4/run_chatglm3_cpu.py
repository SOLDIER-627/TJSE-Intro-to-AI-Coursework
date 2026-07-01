from transformers import AutoTokenizer, AutoModel
import torch
import time
import warnings


class ChatGLM3:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.history = []
        self.load_model()

    def load_model(self):
        """安全加载模型"""
        print("正在初始化ChatGLM3...")
        start_time = time.time()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            # 内存优化配置
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).eval()
            print(f"初始化完成，耗时 {time.time() - start_time:.1f}s")
            print("输入对话内容开始交流，输入'exit'结束\n")

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            exit(1)

    def chat(self, query):
        try:
            # 修复后的生成接口
            response, self.history = self.model.chat(
                self.tokenizer,
                query,
                history=self.history,
                temperature=0.7,
                top_p=0.9,
            )
            return response
        except Exception as e:
            return f"[生成错误] {str(e)}"


def main():
    # 配置项
    model_path = "/mnt/workspace/chatglm3-6b"
    # 初始化对话系统
    chatbot = ChatGLM3(model_path)
    # 对话循环
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
            # 流式输出
            response = chatbot.chat(user_input)
            for chunk in [response[i:i + 50] for i in range(0, len(response), 50)]:
                print(chunk, end="", flush=True)
                time.sleep(0.05)
            print(f"\n[响应耗时: {time.time() - start_time:.1f}s]")
        except KeyboardInterrupt:
            print("\n提示: 输入'退出'结束程序")
        except Exception as e:
            print(f"系统错误: {str(e)}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()