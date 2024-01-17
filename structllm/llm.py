import openai 
import time
import requests
import openai

class gpt(object):
    def __init__(self, model, key):
        self.model = model
        openai.api_key = key
    
    def get_response(self, prompt, flag=0, num=1):
        if self.model=="text-davinci-003" or type(prompt)==str:
            print("text-davinci-003")
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=800,
                temperature=0.5,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=1,
                # stop=stop_sequences,
                logprobs=1,
                n=1,
                best_of=1,
            )
            return response["choices"][0]['text']
        
        else:
            start_time = time.time()
            while True:
                try:
                    if time.time() - start_time > 150:  # 300 seconds = 5 minutes
                        raise TimeoutError("Code execution exceeded 5 minutes")

                    response = openai.ChatCompletion.create(
                        # model="gpt-3.5-turbo",
                        model=self.model,
                        messages=prompt,
                        temperature=1,
                        max_tokens=2048,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n=num
                    )
                    if num == 1 and (not flag):
                        return response.choices[0].message["content"]
                    else:
                        return response.choices
                
                except requests.exceptions.RequestException as e:
                    # 如果发生网络异常，等待10秒后重试
                    print(f"Network error: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)

                except openai.error.OpenAIError as e:
                    # 捕获并处理OpenAI API异常
                    if "maximum context length" in str(e):
                        raise ValueError(e)
                    
                    print(f"An OpenAI API error occurred: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)

                except openai.error.RateLimitError as e:
                    err_mes = str(e)
                    if "You exceeded your current quota" in err_mes:
                        print("You exceeded your current quota")
                    print('openai.error.RateLimitError\nRetrying...')
                    time.sleep(30)

                except openai.error.ServiceUnavailableError:
                    print('openai.error.ServiceUnavailableError\nRetrying...')
                    time.sleep(20)

                except openai.error.Timeout:
                    print('openai.error.Timeout\nRetrying...')
                    time.sleep(20)

                except openai.error.APIError:
                    print('openai.error.APIError\nRetrying...')
                    time.sleep(20)

                except openai.error.APIConnectionError:
                    print('openai.error.APIConnectionError\nRetrying...')
                    time.sleep(20)
                
                except TimeoutError as e:
                    # Handle the custom TimeoutError exception
                    print(f"Code execution exceeded 5 minutes: {e}")
                    # Optionally, you can re-raise the exception to terminate the script
                    raise e