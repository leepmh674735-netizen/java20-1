# 디코더를 사용해 새로운 이미지 생성하기
mins, maxs - np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predicr(sample)