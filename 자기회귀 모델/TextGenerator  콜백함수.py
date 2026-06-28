import numpy as np
from tensorflow.keras import callbacks

class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        super().__init__()
        self.index_to_word = index_to_word
        # 오타 수정: index, word 변수 사이 쉼표 추가
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    # 텍스트 생성의 다양성을 조절하는 템퍼러처(Temperature) 샘플링
    def sample_from(self, probs, temperature):
        # 0에 가까울수록 가장 확률이 높은 단어만 선택(결정론적)
        # 1에 가까울수록 원래 확률 분포대로 선택(다양성 증가)
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature): 
        # 시작 프롬프트를 공백 기준으로 쪼개 토큰 ID로 변환 (기본값 1은 OOV 토큰 가정)
        start_tokens = [       
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        
        # 최대 토큰 길이에 도달하거나, 문장 종료 토큰(0이라 가정)이 나올 때까지 반복
        while len(start_tokens) < max_tokens and sample_token != 0:
            # 1. 입력 데이터를 모델 셰이프에 맞게 2차원 배열로 변환 (Batch 크기 = 1)
            x = np.array([start_tokens])
            
            # 2. [누락 수정] 현재 학습 중인 모델로 다음 단어의 확률 분포 예측
            # verbose=0을 주어야 에포크 도중 예측 로그가 화면을 가리지 않습니다.
            y = self.model.predict(x, verbose=0)
            
            # 3. 모델 출력의 가장 마지막 타임스텝([-1])의 확률 분포를 가져와 샘플링
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            
            # 4. 정보 기록 및 프롬프트 업데이트
            info.append({'prompt': start_prompt, 'word_probs': probs})
            start_tokens.append(sample_token)
            
            # 단어 사전 크기를 벗어나는 에러 방지를 위해 예외 처리 추가
            if sample_token < len(self.index_to_word):
                next_word = self.index_to_word[sample_token]
            else:
                next_word = "[Unknown]"
                
            start_prompt = start_prompt + ' ' + next_word
            
        print(f"\n[실시간 생성 텍스트] \n{start_prompt}\n")
        return info
    
    # 에포크가 끝날 때마다 자동으로 실행되는 케라스 콜백 메서드
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- 에포크 {epoch+1} 종료: 텍스트 생성 테스트 ---")
        self.generate("recipe for", max_tokens=100, temperature=1.0)