import json

# 1. 파일 경로 지정 및 데이터 로드
file_path = '/app/data/epirecipes/full_format_recipes.json'

with open(file_path, 'r', encoding='utf-8') as json_data:
    recipe_data = json.load(json_data)

# 2. 리스트 컴프리헨션을 이용한 데이터 필터링 및 텍스트 포맷팅
filtered_data = [
    f"Recipe for {x['title']} | {' '.join(x['directions'])}"
    for x in recipe_data
    if 'title' in x 
    and x['title'] is not None 
    and 'directions' in x 
    and x['directions'] is not None
]

# 3. 데이터 검증 출력 (선택 사항)
print(f"총 {len(filtered_data)}개의 유효한 레시피가 추출되었습니다.")
if filtered_data:
    print("\n--- 첫 번째 데이터 샘플 ---")
    print(filtered_data[0][:200] + "...") # 너무 길 경우를 대비해 앞부분만 출력