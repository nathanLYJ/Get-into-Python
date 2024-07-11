# Day-10

1. 정적 메서드 (`@staticmethod`):
    - 정적 메서드는 기본적으로 클래스 내부에 정의된 일반 함수와 유사합니다.
    - 기본 메서드와의 주요 차이점은 다음과 같습니다:
        - 클래스나 인스턴스의 상태에 접근하거나 수정하지 않습니다.
        - 클래스나 인스턴스를 통해 호출할 수 있지만, 첫 번째 인자로 self나 cls를 받지 않습니다.
    - 주로 클래스와 관련은 있지만, 클래스나 인스턴스의 상태와는 독립적인 기능을 구현할 때 사용합니다.
2. 클래스 메서드 (`@classmethod`):
    - 개념 설명
        - `__init__`에 있는 속성뿐만 아니라, 클래스 레벨의 속성(클래스 변수)을 조작할 때 주로 사용됩니다.
        - 클래스 전체에 대한 동작을 정의하는 데 사용되는 것은 맞습니다.
    - 주요 특징:
        - 첫 번째 인자로 클래스 자체를 받습니다 (관례상 `cls`로 명명).
        - 인스턴스를 생성하지 않고도 클래스 레벨에서 호출할 수 있습니다.
        - 주로 대체 생성자를 만들거나 클래스 전체에 영향을 미치는 메서드를 정의할 때 사용합니다.

```py
# 1. 동물들에 대한 이름, 동물의 타입, 배고픔, 행복도
# 다마코치 프로그램!

class VirtualPet: 
 def __init__(self, name, animal_type): # 사용자 입력 받을 내용
  self.name = name
  self.animal_type = animal_type
  self.hunger = 50
  self.happiness = 50
  self.energy = 100
 def status(self):
  return f"{self.name}의 상태 : 배고픔{self.hunger}, 행복도 {self.happiness} 입니다"
 # 먹이주기 메서드
 def feed(self):
  self.hunger -= 10
  self.happiness += 10
  print(f'{self.name}에게 먹이를 주었습니다.')
 
 # 빗질
 def brush(self):
  self.happiness += 15
  self.energy -= 5
  print(f'{self.name}에게 빗질해 에너지가{self.energy} 되었습니다')

 
my_pet = VirtualPet("하루","스피츠")
my_pet.status()
my_pet.feed()
print(my_pet.status())
my_pet.brush()
print(my_pet.status())

```

```py

class QuizGame:
 # 문제와 답이 있고, 미리 정해진 답을 유저가 맞추면 득점 / 못맞추면 실점/점수부여 X
 def __init__(self):
  self._questions = {
   "내 이름은?":"유원길",
   "내 성별은?":"남자"
  }
  self._score = 0

 def play(self):
  for question, answer in self._questions.items():
   # 1. 이미 만들어진 질문을 불러오기
   user_answer = input(question + " ")
   # 2. 입력이 된값과 실제 답에 대해서 비교
   if user_answer.lower() == answer.lower():
    print(f"정답:{user_answer}")
    self._score += 1
   else:
    print(f"오답:{user_answer}")

  print(f"정수 : {self._score} 입니다.")
 
 def get_score(self):
  return self._score
 
 def add_question(self, question, answer):
  if question not in self._questions:
   self._questions[question] = answer
   print("질문이 추가되었습니다.")
  else:
   print("이미 존재하는 질문이입다.")


game = QuizGame()
game.add_question("내 이름은?","유원길")
game.add_question("내 직업은?","백엔드 개발자")
game.play()

```

```py
# 간단한 RPG 게임
from abc import ABC, abstractmethod

class Character(ABC):
    def __init__(self, name, level=1, health=100):
        self.name = name
        self.level = level
        self.health = health
        self.str = 10

    def level_up(self):
        self.level += 1
        self.str += 10
        print(f"{self.name}의 레벨이 {self.level}로 올랐습니다!")

    def show_status(self):
        print(f"이름: {self.name}, 레벨: {self.level}, 체력: {self.health}")

    @abstractmethod
    def attack(self):
        pass

class Warrior(Character):
    def __init__(self, name, level=1, health=100, attack_power=10):
        super().__init__(name, level, health)
        self.attack_power = attack_power

    def attack(self):
        print(f"{self.name} 전사가 {self.attack_power}의 힘으로 검을 휘둘렀습니다!")

class Mage(Character):
    def __init__(self, name, level=1, health=80, mana=50):
        super().__init__(name, level, health)
        self.mana = mana

    def attack(self):
        mana_used = 10
        self.mana -= mana_used
        print(f"{self.name} 마법사가 {mana_used}의 마나를 사용하여 화염구를 던졌습니다!")

# 객체 생성 및 사용 예시
hero = Warrior("히어로")
mage = Mage("메이지")

# 전사 캐릭터 사용
hero.show_status()
hero.attack()
hero.level_up()
hero.show_status()

print("\n")

# 마법사 캐릭터 사용
mage.show_status()
mage.attack()
mage.level_up()
mage.show_status()
```

```py
class Student:
    school_name = "파이썬 고등학교"  # 클래스 변수

    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
        self.scores = []

    def add_score(self, subject, score):
        self.scores.append({"subject": subject, "score": score})

    def get_average(self):
        if not self.scores:
            return 0
        return sum(item["score"] for item in self.scores) / len(self.scores)

    @staticmethod
    def grade_to_letter(score):
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    @classmethod
    def change_school_name(cls, new_name):
        cls.school_name = new_name
        print(f"학교 이름이 {new_name}으로 변경되었습니다.")

# 사용 예시
student1 = Student("Alice", 10)
student1.add_score("수학", 85)
student1.add_score("영어", 92)

print(f"{student1.name}의 평균 점수: {student1.get_average()}")
print(f"등급: {Student.grade_to_letter(student1.get_average())}")

print(f"현재 학교 이름: {Student.school_name}")
Student.change_school_name("파이썬 과학 고등학교")
print(f"변경된 학교 이름: {Student.school_name}")
```

```py
from abc import ABC, abstractmethod

# 추상 클래스 정의
class Animal(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def make_sound(self):
        pass
    
    def introduce(self):  # 여기를 수정했습니다: introdue -> introduce
        print(f'저는 {self.name}입니다.')
        self.make_sound()

class Dog(Animal):
    def make_sound(self):
        print("멍멍")

poppy = Dog("바둑이")
poppy.introduce()
```

```py
class Example:
 def __init__(self):
  self.public = "누구나 접근 가능"
  self._protrcted = "클래스 내부나 하위 클래스에서 사용 권장"
  self.__private = "클래스 내부에서만 사용 권장"
 
 def get_private(self):
  return self.__private

e = Example()
print(e.public) # 정상
print(e._protrcted) # 경고
print(e.get_private())# 정상 작동
print(e._Example__private) # 네임 맹클링 통한 접근
print(e.__private) # AttributeError 발생
```

```py
class TV(object):
  def __init__(self, size, year, company):
    self.size = size
    self.year = year
    self.company = company

  def describe(self):
    print(self.company + "에서 만든" + self.year + "년형" + self.size + "인치 TV")

class Laptop(TV):
  def describe(self):
    print(self.company + "에서 만든"  + self.year + "년형" + self.size + "인치 노트북")

LG_TV = TV("32", "2022", "LG")
LG_TV.describe()

samsung_microwave = Laptop("15", "2023", "Samsung")
samsung_microwave.describe()


#1. LG에서 만든 2022년형15인치 TV
#   Samsung에서 만든2023년형32인치 노트북

#2. LG에서 만든 2022년형15인치 TV
#   Samsung에서 만든 2023년형 15인치 노트북

#3. LG에서 만든 2022년형 32인치 TV
#   Samsung에서 만든 2023년형 32인치 노트북

#4. LG에서 만든 2022년형 32인치 TV
#   Samsung에서 만든 2023년형 15인치 노트북

#5. LG에서 만든 2023년형 32인치 TV
#   Samsung에서 만든 2022년형 15인치 노트북
```

```py
class Company:
 def __init__(self):
  self.work = True
  self.name = 'Jane'
  self.company_name = 'woman'
 
 def retire(self):
  self.work = False

class Employee(Company):
 def __init__(self,name,company_name):
  super().__init__()
  self.name = name
  self.company_name = company_name

 def introduce(self):
  if self.work == True:
   print('취업 성공')
   print(f'이름은 {self.name} 입니다.')
   print(f'회사는 {self.company_name} 입니다')

  if self.work == False:
   print(f'직업을 구하는 중..')


```

# Day-11

### 판다스

```py
# 넘파이 설치 (명령 프롬프트에서 실행)# pip install numpy# 판다스 설치# pip install pandas# 파이썬 스크립트에서 임포트
import numpy as np
import pandas as pd

print("NumPy 버전:", np.__version__)
print("Pandas 버전:", pd.__version__)

# NumPy를 사용한 배열 생성
np_array = np.array([1, 2, 3, 4, 5])
print("NumPy 배열:", np_array)

# Pandas를 사용한 Series 생성
# pandas series -> 1. 열로 정리 된다, index가 2. 자동으로 생성된다
pd_series = pd.Series([1, 2, 3, 4, 5])
print("Pandas Series:\n", pd_series)

# Pandas를 사용한 DataFrame 생성
# 딕셔너리 형태의 데이터를 DataFrame 메서드에 넣으면 엑셀파일과 같은 데이터 구조가 생선된다.
data = pd.DataFrame({
    '이름': ['김철수', '박영희', '모두연'],
    '나이': [25, 28, 32],
    '직업': ['개발자', '디자이너', '마케터']
})
print("Pandas DataFrame:\n", df)
df = pd.DataFrame(data)
print(df)
print(df.info())
print(df.shape)

# 2. 조건을 만족하는 데이터 필터링

modu_df = df[df['이름'] == '모두연']
print(modu_df)

```

```py
#예시 데이터프레임 생성
data = {
    'Name': ['Alice', 'Bob'],  # 문자열 타입
    'Age': ['25', '30'],       # 숫자 데이터를 문자열로 표현
    'Member': ['True', 'False'],  # 불린 데이터를 문자열로 표현
    'Join Date': ['2021-01-01', '2021-07-01']  # 날짜 데이터를 문자열로 표현
}

df = pd.DataFrame(data)

#데이터 타입 변환
df['Age'] = df['Age'].astype(int)  # 'Age' 열을 정수형으로 변환
df['Member'] = df['Member'].astype(bool)  # 'Member' 열을 불린형으로 변환
df['Join Date'] = pd.to_datetime(df['Join Date'])  # 'Join Date' 열을 datetime으로 변환

#결과 출력
print(df)
print(df.dtypes)
```

```py
import pandas as pd

#시리즈 생성
names = pd.Series(['홍길동', '임꺽정', '이순신'])

#시리즈 출력
print("시리즈 출력:")
print(names)

#시리즈의 데이터 타입 확인
print("\n시리즈의 데이터 타입:")
print(names.dtype)

#시리즈의 인덱스 확인
print("\n시리즈의 인덱스:")
print(names.index)
#사용자 정의 인덱스 설정
names_with_index = pd.Series(['홍길동', '임꺽정', '이순신'], index=['a', 'b', 'c'])

#시리즈 출력
print("사용자 정의 인덱스를 가진 시리즈:")
print(names_with_index)

#특정 인덱스를 사용하여 데이터 접근
print("\n인덱스 'b'의 데이터:")
print(names_with_index['b'])

#벡터화 연산 예시: 모든 이름에 '님'을 추가
print("\n벡터화 연산 예시 (모든 이름에 '님' 추가):")
print(names_with_index + '님')

#나이 데이터 시리즈 생성
ages = pd.Series([30, 35, 40], index=['홍길동', '임꺽정', '이순신'])

#시리즈 출력
print("나이 데이터 시리즈:")
print(ages)

#조건에 따른 데이터 필터링 (나이가 35 이상인 데이터)
print("\n나이가 35 이상인 데이터:")
print(ages[ages >= 35])

```

### 실습 데이터 처리

```py
# 데이터 불러오기

# 구분자 csv = , 기존의 열의 이름을 사용하지 않겠다.
df = pd.read_csv('nasa_http_sample.csv', sep = ',', header = None, 
                 names = ['ip', 'dummy1', 'dummy2', 'timesteamp', 'request', 'status', 'size'])
df = df.drop(['dummy1','dummy2'], axis = 1)
print(df)
# 필터링 해서 출력
print(df.loc[df['status'] == '200'])
print(df.loc[df['ip'] == 'burger.letters.com'])
# 특정 열에 있는 특정 행을 추출!
# start stop step
print(df.loc[1:3:2,'ip'])
#df.loc[행에 대한 조건, 열 대한 조건]
print(df.loc[df['ip'] == '205.212.115.106','request'])
print(df.loc[df['status'] == '404', ['ip','timesteamp']])
```

```py
import pandas as pd

# 1. 실습: 로그 데이터의 기본 정보 확인하기

# a. 데이터 불러오기
print("a. 데이터 불러오기")
df = pd.read_csv('online_shop_orders.csv')

print("\n데이터 미리보기:")
print(df.head())

# b. 기본 정보 확인
print("\nb. 기본 정보 확인")
print("\n데이터 구조:")
print(df.shape)

print("\n수치형 데이터 기술 통계:")
print(df.describe())

print("\n열 이름과 데이터 타입:")
print(df.dtypes)

# c. loc[]와 iloc[]를 사용한 데이터 탐색
print("\nc. loc[]를 사용한 데이터 탐색")
print(df.loc[df['order_date'] == '2023-07-01 09:15:00']) ## ??

# loc[]로 특정 열 선택하여 확인
print("\n주문 ID와 주문 날짜 열 추출")
print(df.loc[:, ['order_id', 'order_date']])

# loc[]를 사용해 조건에 맞는 데이터 선택
print("\n총액이 50달러 이상인 주문:")
print(df.loc[df['total'] >= 50])
```

```py

import pandas as pd

# 주문 데이터를 데이터프레임으로 생성
order_data = {
    'order_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'customer_id': ['C001', 'C002', 'C003', 'C001', 'C004', 'C002', 'C005', 'C003', 'C001', 'C004'],
    'order_date': ['2023-07-01', '2023-07-01', '2023-07-02', '2023-07-02', '2023-07-03', '2023-07-03', 
                   '2023-07-04', '2023-07-04', '2023-07-05', '2023-07-05'],
    'product_category': ['Electronics', 'Books', 'Clothing', 'Electronics', 'Books', 'Clothing', 
                         'Electronics', 'Books', 'Clothing', 'Electronics'],
    'quantity': [2, 3, 1, 1, 2, 4, 1, 2, 3, 1],
    'price': [500, 20, 50, 800, 15, 30, 600, 25, 40, 700],
    'total': [1000, 60, 50, 800, 30, 120, 600, 50, 120, 700]
}

df_orders = pd.DataFrame(order_data)

csv_file_path_orders = "online_shopping_orders.csv"
df_orders.to_csv(csv_file_path_orders, index=False)
# 그룹화 집계
daily_sales = df_orders.groupby('product_category')['total'].sum()
print(daily_sales)
# 그룹화 집계
daily_sales_new = df_orders.groupby('product_category').agg({
 'customer_id':  ['sum'],
 'total' : ['sum']
})
print(daily_sales_new)
```

```py
# 1
daily_sales_new3 = df_orders['quantity'].sum()
print(daily_sales_new3)
#2
daily_sales_new4 = df_orders.groupby('customer_id')['total'].agg(['sum', 'mean'])
print(daily_sales_new4)
#3
daily_sales_new5 = df_orders.groupby('product_category').agg({
 'order_id':['count'],
 'quantity':['mean']
})
print(daily_sales_new5)
```

```py
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': [1, 2, 3, 4]})

df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'],
                    'value': [5, 6, 7, 8]})

merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)
merged_df1 = pd.merge(df1, df2, on='key', how='right')
print(merged_df1)
merged_df2 = pd.merge(df1, df2, on='key', how='left')
print(merged_df2)
```

# Day-12

```py
import pandas as pd

s = pd.Series(['A','B','C','C','C','C','B','B','B'])

count = s.value_counts()
print(count)
# - `normalize`: 빈도 대신 비율을 반환합니다 (기본값: False).
# - `sort`: 결과를 정렬합니다 (기본값: True).
# - `ascending`: 오름차순으로 정렬합니다 (기본값: False).
# - `dropna`: NaN 값을 제외합니다 (기본값: True).
```

```py
# 비율로 표시하는 normalize
count = s.value_counts(normalize = True)
print(count)
# 비율로 표시하는 normalize
count = s.value_counts(ascending = True)
print(count)
```

```py
df = pd.DataFrame({
    'Category' : ['A', 'B', 'C' , 'C' , 'C' , 'C' ,'C', 'B' , 'B' , 'B'],
    'Value' : [10, 20, 30 , 40 , 50 , 60 , 70, 80 , 90 , 100]
})
category_counts = df['Category'].value_counts()
print(category_counts)
```

```py
# NaN 값 포함하기
import pandas as pd

s = pd.Series(['A', 'B', 'C' , 'C' , None, None, None, 'B' , 'B' , 'B'])
count = s.value_counts()
print(count)
# dropna -> null값을 drop 시켜라 -> 제거하라
count_with_nan = s.value_counts(dropna = False)
print(count_with_nan)
```

```py
# apply 함수

import pandas as pd

s = pd.Series([1,2,3,4,5])

# def square(x):
#     return x**2
# squared = s.apply(square)
squared = s.apply(lambda x: x*3)


print(squared)
```

```py
import pandas as pd

# DataFrame 생성
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 각 요소에 2를 더하는 함수 정의
def add_two(x):
    return x + 2

# apply()를 사용하여 열 단위로 함수 적용
df_applied = df.apply(add_two)

print(df_applied)
```

```py
s = pd.Series([1,2,3,4,5])

def square(x) :
    return x**2

squared = s.apply(square)
print(squared)

print(square(s))
```

```py
import numpy as np

df = pd.DataFrame({
    '값': [1, 2, 3, 4, 5],
    '함수': ['제곱', '세제곱', '제곱근', '로그', '사인']
})

def 동적_함수(row):
    if row['함수'] == '제곱':
        return row['값'] ** 2
    elif row['함수'] == '세제곱':
        return row['값'] ** 3
    elif row['함수'] == '제곱근':
        return np.sqrt(row['값'])
    elif row['함수'] == '로그':
        return np.log(row['값'])
    elif row['함수'] == '사인':
        return np.sin(row['값'])

df['결과'] = df.apply(동적_함수, axis=1)
print(df)
```

```py
import matplotlib.pyplot as plt

# 데이터 준비
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 그래프 생성
plt.plot(x, y)

# 제목 추가
plt.title('간단한 선 그래프')

# x축 레이블
plt.xlabel('X 축')

# y축 레이블
plt.ylabel('Y 축')

# 그래프 표시
plt.show()
```

```py
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 사용 가능한 스타일 목록 출력
print("사용 가능한 스타일:")
print(plt.style.available)

# 스타일 설정 (선택적)
# plt.style.use('default')  # 기본 스타일 사용
# plt.style.use('classic')  # 클래식 스타일 사용
# 또는 다른 사용 가능한 스타일 중 하나를 선택

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

# 선 스타일, 색상, 마커 설정
plt.plot(x, y, linestyle='--', color='r', marker='o')

plt.title('커스터마이즈된 그래프', fontsize=20)
plt.xlabel('X 축', fontsize=14)
plt.ylabel('Y 축', fontsize=14)

# 격자 표시
plt.grid(True)

plt.show()

```

```py
import matplotlib.pyplot as plt

x = range(1, 6)
y1 = [1, 4, 3, 7, 5]
y2 = [2, 5, 3, 4, 6]

plt.figure(figsize=(12, 5))

# 2x1 구조의 서브플롯 생성
plt.subplot(1, 2, 1)
plt.plot(x, y1, 'r-')
plt.title('첫 번째 그래프')

plt.subplot(1, 2, 2)
plt.plot(x, y2, 'b--')
plt.title('두 번째 그래프')

plt.tight_layout()
plt.show()
```

```py
# 사용자 유형별 접속 데이터
user_data = {
    '사용자 유형': ['일반', '프리미엄', 'VIP'],
    '접속 횟수': [500, 300, 200]
}

# 시간대별 접속 데이터
time_data = {
    '시간': list(range(24)),
    '접속 횟수': [10, 8, 5, 3, 2, 5, 15, 30, 50, 70, 80, 90, 85, 75, 60, 65, 70, 55, 40, 35, 25, 20, 15, 12]
}

# DataFrame 생성
user_df = pd.DataFrame(user_data)
time_df = pd.DataFrame(time_data)
```

```py
plt.figure(figsize=(8, 8))  # 8x8 크기의 그림 생성
user_df.plot(kind='pie', y='접속 횟수', labels=user_df['사용자 유형'], autopct='%1.1f%%')
plt.title('사용자 유형별 접속 비율')  # 제목 설정
plt.ylabel('')  # y축 레이블 제거
plt.show()  # 그래프 표시

print(log_df.head())
print(user_df.head())
print(log_df.describe())
merged_df1 = pd.merge(log_df, user_df, on='ip_address', how='left')
print(merged_df1)
merged_df1_count = merged_df1['user_type'].value_counts(normalize=True)
print(merged_df1_count)
merged_df1_mean = merged_df1.groupby('user_type')['ip_address'].count().mean()
print(merged_df1_mean)
```

# Day-13

### 모듈 외부 코드 끌어와서 사용하는 방식

- 내PC 설치한 모듈 리스트 불러오기 {!pip list}
  - 2.0 설치시 pip, 3.0 부터 pip3
- 패키지는 from, 여러 모듈의 묶음.

#### 모듈 양식

- import 모듈1
- import 모듈2, 모듈3
- import 모듈4 as 별칭
- from 폴더_또는_파일이름(모듈) import 파일_또는_변수명
- from 폴더_또는_파일이름(모듈) import 변수 as 별칭2

- 모듈1.변수
- 모듈1.함수()
- 모듈1.클래스()
- 별칭.변수
- 별칭.함수()

```py
# Faker모듈 설치
from faker import Faker
fake = Faker('ko-KR')
print(fake.name())
print(fake.address())
print(fake.postcode())  #  우편 번호
print(fake.country())  #  국가명
print(fake.company())  #  회사명
print(fake.job())   # 직업명
print(fake.phone_number())    #휴대 전화 번호
print(fake.email())    #이메일 주소
print(fake.user_name()) #   사용자명
print(fake.pyint(min_value=0, max_value=100)) #  0부터 100 사이의 임의의 숫자
print(fake.ipv4_private())  #  IP 주소
print(fake.text())    #임의의 문장 (한글 임의의 문장은 fake.catch_phrase() 사용)
print(fake.color_name())    #색상명
```

```py
import os

# 올바른 경로 지정 방법
os.chdir(r"C:\Users\Nathan\Desktop\python")

# 또는 슬래시 사용
# os.chdir("C:/Users/Nathan/Desktop/python")

# 현재 작업 디렉토리 확인
print(os.getcwd())

import theater_module as tmt
```

```py
# 예제 1: import 문 사용
import travel_pkg.domestic.seoul

trip = travel_pkg.domestic.seoul.SeoulPkg()
trip.detail()

try: #예외가 발생할것 같은 구문
  num = int(input("숫자를 입력해주세요"))

except: #예외가 발생했을때 출력하는 구문
  print("숫자가 입력되지 않았습니다.")

else: #예외가 발생하지 않았을때 문장 실행
  print(f"입력된 숫자는 {num} 입니다.")

finally: #무조건 실행
  print("이 구문은 무조건 실행됩니다.")
```

```py
sentence = list("hello world")

while (len(sentence) + 1):
  try:
    print(sentence.pop(0))
  except Exception as e:
    print(e)
    break
```

```py
try:
    x = int(input("숫자를 입력하세요: "))
    result = 10 / x
    print(result)
except ValueError:
    print("유효한 숫자를 입력해야 합니다.")
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except:
    print("알 수 없는 오류가 발생했습니다.")
```

```py
# 연습문제 3
import random
answer = random.randint(1, 10)

def guess_number(answer):
  try:
    guess = int(input("숫자를 넣어주세요 : "))
    if answer == guess:
      print("정답 !")
    else:
      print("오답 !")
  except ValueError:
    print("숫자가 아닙니다.")

guess_number(answer) #hello를 입력한다면?

for i in range(3):
  try:
    print(i, 3//i)
  except ZeroDivisionError:
    print("Not divided by 0")
```

1. try-except 문의 동작 방식에 대해 설명드리겠습니다.
2. try는 에러 이름이나 True/False 값으로 판단하는 것이 아닙니다. try-except의 동작 원리는 다음과 같습니다:

#### 예외 발생 감지

- try 블록 내의 코드를 실행하는 동안 발생하는 모든 예외를 감지합니다.
- 예외가 발생하지 않으면 try 블록의 코드가 정상적으로 모두 실행됩니다.

#### 예외 처리

- 예외가 발생하면, 즉시 try 블록의 실행을 중단하고 해당 예외와 일치하는 except 블록으로 이동합니다.
- except 블록에서 지정한 예외 타입과 발생한 예외의 타입이 일치하면 해당 except 블록이 실행됩니다.

#### 예외 타입 매칭

- except 문에 명시적으로 예외 타입을 지정할 수 있습니다. 예: except ValueError:
- 여러 except 블록을 사용하여 다양한 예외 타입을 처리할 수 있습니다.
- 예외 타입을 지정하지 않으면 (except:), 모든 타입의 예외를 잡습니다.

### 크롤링

##### URL 구조

```js
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              href                                              │
├──────────┬──┬─────────────────────┬────────────────────────┬───────────────────────────┬───────┤
│ protocol │  │        auth         │          host          │           path            │ hash  │
│          │  │                     ├─────────────────┬──────┼──────────┬────────────────┤       │
│          │  │                     │    hostname     │ port │ pathname │     search     │       │
│          │  │                     │                 │      │          ├─┬──────────────┤       │
│          │  │                     │                 │      │          │ │    query     │       │
"  https:   //    user   :   pass   @ sub.example.com : 8080   /p/a/t/h  ?  query=string   #hash "
│          │  │          │          │    hostname     │ port │          │                │       │
│          │  │          │          ├─────────────────┴──────┤          │                │       │
│ protocol │  │ username │ password │          host          │          │                │       │
├──────────┴──┼──────────┴──────────┼────────────────────────┤          │                │       │
│   origin    │                     │         origin         │ pathname │     search     │ hash  │
├─────────────┴─────────────────────┴────────────────────────┴──────────┴────────────────┴───────┤
│                                              href                                              │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- GET /index.html HTTP/1.1

- user-agent: MSIE 6.0; Windows NT 5.0
- accept: text/html; */*
- cookie: name = value
- referer: <http://www.naver.com>
- host: <www.paullab.co.kr>

1. **Request line**: 데이터 처리 방식, 기본 페이지, 프로토콜 버전 등을 나타냅니다.
2. **User-Agent**: 사용자 웹 브라우저 종류 및 버전 정보를 나타냅니다.
3. **Accept**: 웹 서버로부터 수신되는 데이터 중 웹 브라우저가 처리할 수 있는 데이터 타입을 의미합니다.

    여기서 `text/html`은 text, html 형태의 문서를 처리할 수 있고,  `*/*`는 모든 문서를 처리할 수 있다는 의미입니다. (이를 MIME 타입이라 부르기도 합니다.)

4. **Cookie**: HTTP 프로토콜 자체가 세션을 유지하지 않는 State-less(접속 상태를 유지하지 않는) 방식이기 때문에 로그인 인증을 위한 사용자 정보를 기억하려고 만든 인위적인 값입니다. 즉, 사용자가 정상적인 로그인 인증 정보를 가지고 있다는 것을 판단하고자 사용합니다.
5. **Referer**: 현재 페이지 접속 전에 어느 사이트를 경유했는지 알려주는 도메인 혹은 URL 정보입니다.
6. **Host**: 사용자가 요청한 도메인 정보입니다.

```py
from IPython.display import HTML
html_sample = 
'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>블로그 소개</title>
</head>
<body>
    <header>
        <h1>환영합니다!</h1>
        <p>여기는 김진환의 블로그입니다.</p>
    </header>

    <main>
        <section>
            <h2>소개</h2>
            <p>안녕하세요, 김진환입니다. 저는 25살이고, 서울대학교에 다니고 있습니다.</p>
        </section>

        <section>
            <h2>내가 좋아하는 것들</h2>
            <ul>
                <li>독서</li>
                <li>여행</li>
                <li>프로그래밍</li>
            </ul>
        </section>

        <section>
            <h2>자주 방문하는 웹사이트</h2>
            <p>제가 자주 방문하는 웹사이트는 <a href="https://www.github.com">GitHub</a>입니다.</p>
        </section>

        <section>
            <h2>내 프로필 사진</h2>
            <img src="https://via.placeholder.com/150" alt="내 프로필 사진">
        </section>
    </main>

    <footer>
        <p>김진환의 블로그에 방문해 주셔서 감사합니다!</p>
    </footer>
</body>
</html>
display(HTML(html_sample))
```

```py
import requests
from bs4 import BeautifulSoup

# 웹 페이지 가져오기
response = requests.get('https://www.naver.com')

# BeautifulSoup 객체 생성
soup = BeautifulSoup(response.text, 'html.parser')

# 페이지 제목 출력
print(soup)
```

```py
import requests
from bs4 import BeautifulSoup

response = requests.get('http://www.paullab.co.kr/stock.html')

response.encoding = 'utf-8' # 인코딩 설정
html = response.text

soup = BeautifulSoup(html, 'html.parser')
print(soup)
```

```py

## BeautifulSoup 이란?

![출처: BeautifulSoup 공식 문서](./image/Untitled.png)

`BeautifulSoup`은 HTML과 XML 문서를 파싱하는 라이브러리입니다. 웹 크롤링을 할 때, HTML 문서에서 원하는 데이터를 추출하는 데 자주 사용됩니다. str 타입의 html 데이터를 html 구조를 가진 데이터로 가공하는 것도 가능합니다
```

```py
soup.a
soup.title
soup.title.string
soup.table
soup.tr
soup.em
```

### find()

```py
import requests
from bs4 import BeautifulSoup

response = requests.get('http://www.paullab.co.kr/stock.html')

response.encoding = 'utf-8' # 인코딩 설정
html = response.text

soup = BeautifulSoup(html, 'html.parser')

soup.find_all('h2')[0:3]
soup.find_all('table', class_='table')[0] # class_ : 예약어
# 예약어: 특정한 기능을 수행하도록 미리 예약되어 있는 것
soup.find('head').find('title')
soup.select('.table > tbody > tr')[2]
oneStep = soup.select('tbody')[0]
oneStep
twoStep = oneStep.select('tbody > tr')[1:] 
twoStep
print(twoStep[0].select('td')[0].text) # 날짜
print(twoStep[0].select('td')[1].text) # 날짜

```

```py
날짜 = []
종가 = []

for i in twoStep:
    날짜.append(i.select('td')[0].text)
    종가.append(int(i.select('td')[1].text.replace(',', '')))

print(날짜)
print(종가)
```

```py
import plotly.express as px
import plotly.io as pio

# 브라우저에서 그래프 열기
pio.renderers.default = "browser"

fig = px.line(x=날짜, y=종가, title='jejucodingcamp')
fig.show()
```

### Day-14

```py
for i in range(10):
    print(i)

# for 반복문 시작
# 범위(0부터 9까지) 안에 i번째 밑에 내용 반복 실행
# 실행 내용은 i 를 출력
# 범위의 마지막 값(9)에 도달하면 반복 종료
# 반복 횟수는 10번
i = iter('hello')
next(i)
```

```py
class MyIterator:
    def __init__(self, stop):
        self.stop = stop

    def __iter__(self):
        self.currentValue = 0
        return self

    def __next__(self):
        if self.currentValue >= self.stop:
            raise StopIteration
        result = self.currentValue
        self.currentValue += 1
        return result

my_iterator = MyIterator(5)

for i in my_iterator:
    print(i)

for i in my_iterator:
    print(i)

# 결국 for는 iter먼저 실행하고, next로 StopIteration
# i = iter(li)
# next(i)
```

```py
class MyIterator:
    def __init__(self, stop):
        self.stop = stop
        self.data = list(range(stop))  # 데이터를 미리 생성하여 저장

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.stop:
            raise StopIteration
        result = self.data[self.current_index]
        self.current_index += 1
        return result

    def __getitem__(self, index):
        if isinstance(index, slice):
            # 슬라이싱 지원
            return [self.data[i] for i in range(*index.indices(self.stop))]
        elif 0 <= index < self.stop:
            return self.data[index]
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        return self.stop

# 사용 예시
my_iterator = MyIterator(5)

# 반복 사용
print("Iteration:")
for i in my_iterator:
    print(i)

# 인덱싱 사용
print("\nIndexing:")
print(my_iterator[2])  # 2 출력
print(my_iterator[1:4])  # [1, 2, 3] 출력

# 길이 확인
print("\nLength:")
print(len(my_iterator))  # 5 출력

# 두 번째 반복도 가능
print("\nSecond iteration:")
for i in my_iterator:
    print(i)
```

```py
'0100101'.replace('0', ' ').replace('1', '#')
'0100101'\
 .replace('0', ' ')\
 .replace('1', '#')
```

### 일급함수와 고차함수

```py
x = 10 # 변수 x에 10을 할당

def f():
    print('hello world')

x = f # 변수 x에 f함수 할당, '일급 함수'는 함수를 마치 값처럼 취급

print(x())
```

```py
def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

x = [add, sub, mul, div] # 값이 들어갈 수 있는 공간에 함수 이름을 다 넣어보는 것!
x[0](1, 2)

def add(a, b):
    return a + b

def hello(f):
    return f(10, 20) + f(20, 30)

hello(add)
```

### 클로저

```py
# 함수를 리턴
def f():
    def ff():
        print('hello')
    return ff

x = f()
x()
# 클로저, 파이썬에서 팩토리 함수라고도 합니다.
# 위 원리를 이용한 것입니다.
def f(x):
    def ff(y):
        return x ** y
    return ff

x = f(3)
# 이 다음부터는 3 ** ? 인데, 3을 변경시킬 수 없습니다.
# def ff(y):
#     return 3 ** y
x(2)

xx = f(4)
# 이 다음부터는 4 ** ? 인데, 4을 변경시킬 수 없습니다.
xx(2)

# point1: 원래 휘발되었어야 하는 메모리 공간이 참조로 인해 살아있게 됩니다.
# point2: 휘발되었어야 하는 공간에 남아있는 변수는 변경 불가능한 변수로 남아있게 됩니다.
# point3: 그리고 이 공간에 접근하는 기술을 클로저라고 합니다.
```

데커레이터

```py
def login(function):
    pass

@login
def 게시판읽기():
    pass
def simple_decorator(function):
    def wrapper():
        print("전")
        # function()
        print("후")
    return wrapper


@simple_decorator
def hello():
    print("Hello, World!")


hello() # 데코레이터가 없는 상태에서는 simple_decorator(hello)() 와 같습니다.
```

```py
# 전처리 작업을 하고 싶다!
# 후처리 작업을 하고 싶다!

def hello():
    pass
# 여기까지만 보셨을 때 데커레이터를 사용하는 이유가 어느정도 이해가 가셨으면 좋겠습니다.
data = [1, '2', 3, '4', 5]

@전처리
def custom_sum(d):
    return sum(d)

print(custom_sum(data))
```

```py
# 여기까지만 보셨을 때 데커레이터를 사용하는 이유가 어느정도 이해가 가셨으면 좋겠습니다.
data = [1, '2', 3, '4', 5]

####### 이부분을 숨길수 있다
def 전처리(function):
    def wrapper(d):
        return function(list(map(int, d)))
    return wrapper
######

@전처리
def custom_sum(d):
    return sum(d)

print(custom_sum(data))

# 하나의 코드에 다 집어넣으면 되는 것 아닌가요? 아는척하는거에요? 너무 어렵게 짰어요!!
# 가독성을 해치는 것 아닌가요?

# 답: 재사용성이 크게 높아집니다.
# 그리고 이 데커레이터가 숨겨졌을 때(추상화 되었을 때) 가독성이 그 전보다 훨씬 뛰어나집니다.
```

### lambda

```py
def f(x):
    return x ** 2

f = lambda x: x ** 2
```

```py
data = [
    [1, 400, 'h'],
    [2, 300, 'he'],
    [4, 100, 'hel'],
    [3, 200, 'hell'],
    [5, 500, 'hello'],
]

def f(x):
    return x[1]

sorted(data, key=f) # lambda에 가장 큰 사용 이유는 재사용하지 않겠다!라는 것입니다.
```

```py
#1차원의 점들이 주어졌을 때, 그 중 가장 거리가 짧은 것의 쌍을 출력하는 함수를 작성하시오. (단 점들의 배열은 모두 정렬되어있다고 가정한다.)

#예를들어 S={1, 3, 4, 8, 13, 17, 20} 이 주어졌다면, 결과값은 (3, 4)가 될 것이다.
s = [1, 3, 4, 8, 13, 17, 20]
ss = s[1:]

list(zip(s, ss))
```

```py
s = [1, 3, 4, 8, 13, 17, 20]
ss = s[1:]

sorted(zip(s, ss), key=lambda x: x[1]-x[0])
# sorted(zip(s, ss), key=lambda x: x[1]-x[0])[0]
```

### Day-15

```py
# 다음중 텍스트에서 숫자만 골라 모두 더하라
# 'h1e2l3lo11wo2r3l9d'

s = 'h1e2l3lo11wo2r3l9d'
sum(map(int, filter(lambda x: x.isdigit(), 'h1e2l3lo11wo2r3l9d')))

# type(s) == 'str'
# isinstance(s, str)
```

### args,kwargs

```py
def print_args(*args):
    print(args) # 출력: (100, True, 'Licat')

print_args(100, True, 'Licat')
```

```py
# 파선아실
def f(a, b, *args):
    print(a, b)
    print(args)

f(1, 2, 3, 'abc', 'def')
```

```py
# 파라미터로 들어갔을 때에는 패킹을 했고
# 아규먼트로 들어갔을 때에는 언패킹을 했습니다.
# 언패킹
def func(a, b, c):
    print(a, b, c)

args = (10, 20, 30)
func(*args)
# (10, 20, 30) => *args => 10, 20, 30
```

```py
a, b, c = range(3)
a
def f(a, **kwargs):
    print(a)
    print(kwargs)

f(100, name='Licat', age='10')
```

```py
data = [
    {
        'name': 'Licat',
        'age': 3
    },
    {
        'name': 'John',
        'age': 7
    },
    {
        'name': 'Jane',
        'age': 4
    }
]
sorted(data, key=lambda x: x['age'], reverse=True)
sorted(data, reverse=True, key=lambda x: x['age'])
```

```py
def f(*args, **kwargs):
    print(args)
    print(kwargs)

f(100, name='Licat', age='10')
f(100, 200, name='Licat', age='10')
# f(100, name='Licat', 200, age='10') #error
```

```py
one, two, *three = 1, 2, 3, 4, 5
print(one, two, three)
```

```py
a, b, *c = 'hello world'
c # 출력: ['l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
```

```py
a = 1,2,3,4,5,6
i,j,*k,l = a
print(i)
print(j)
print(k)
print(l)
```

```py
# 파선아실
def f(a, b, *args,):
    print(a, b)
    print(args)
    print(c)
f(1, 2, 3, 'abc', 'def',4)
```

```py
# 필
def g():
    yield '홀'
    yield '짝'
    yield '홀'
    yield '짝'
    yield '홀'
    yield '짝'
    yield '홀'
    yield '짝'
    yield '홀'
    yield '짝'
    yield '홀'
    yield '짝'
list(zip([1, 2, 3, 4, 5, 6], g()))
```

```py
class MyIterator:
    def __init__(self, stop):
        self.current_value = 0  # 현재 값
        self.stop = stop  # 순회를 멈출 값

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_value >= self.stop:
            raise StopIteration
        result = self.current_value
        self.current_value += 1
        return result

my_iterator = MyIterator(5)

for i in my_iterator:
    print(i)

for i in my_iterator:
    print(i)

for i in my_iterator:
    print(i)
```

```py
class MyIterator:
    def __init__(self, stop):
        self.stop = stop

    def __iter__(self):
        self.current_value = 0  # __iter__에서 초기화
        return self

    def __next__(self):
        if self.current_value >= self.stop:
            raise StopIteration
        result = self.current_value
        self.current_value += 1
        return result

my_iterator = MyIterator(5)

for i in my_iterator:
    print(i)
```

#### 제너레이터는 이터레이터를 생성하는 함수이다

- **iter**() 와 **next**() 메서드를 구현합니다.
- next() 함수를 사용하여 다음 요소를 가져올 수 있습니다.
- 모든 요소를 순회한 후에는 StopIteration 예외를 발생시킵니다.

1. 제너레이터는 휘발성있는 함수, yield 한번 순회하면 소진되어 재사용 불가능

### nonlocal

```py
# 다음과 같은 자료형이 있습니다. 
# A학점 10명
# B학점 15명
# C학점 20명
# F학점 나머지

import random as r

data = [r.randint(0, 100) for i in range(100)]
data

# 이 데이터를 가지고 아래와 같은 형태를 만들어주세요.
# 제너레이터를 활용해주세요.

# data = [
#     [37, 'A'], 
#     [27, 'B'], 
#     [17, 'C'], 
#     ]
def g():
    for i in range(10):
        yield 'A'
    for i in range(15):
        yield 'B'
    for i in range(20):
        yield 'C'
    while True:
        yield 'F'

list(zip(sorted(data, reverse=True), g()))
```

```py
# nonlocal
a = 10
def f():
    a = 100
    print(f'f a: {a}')
    def ff():
        a = 1000
        print(f'ff a: {a}')
        def fff():
            global a  # global a nonlocal a로 변경해보세요.
            a = 10000
            print(f'fff a: {a}')
        fff()
        print(f'ff a: {a}')
    ff()
f()
print(f'global a: {a}')
```

```py
class A:
 def __str__(self):
  return 'hello'
 def __repr__(self):
  return 'world'
a = A()
print(a)
a
```

```py
f = open('student.csv', 'w')
l = ['licat', 'mura', 'binky']
s = ''
for i, j in enumerate(l, 1):
    s += f'{i}번 {j}입니다.'
f.write(s)
f.close()
```

```py
f = open('index.html', 'w')
s = '''<html>
<head>
</head>
<body>
<h1>hello world</h1>
</body>
</html>
'''
f.write(s)
f.close()
```

```py
# 하나의 코드로 합친 것
# 데이터를 어디에다 저장해야 할까?
# 리스트, 튜플, 딕셔너리, 셋, 별도 클래스
# 확정성 고려해서: 리스트 안에 별도 클래스로 저장이 가장 적절함
import requests
from bs4 import BeautifulSoup

url = 'https://www.flaticon.com/free-animated-icon/money-bag_6172509'
data = requests.get(url)
soup = BeautifulSoup(data.text, 'html.parser')


result = []

for i, img in enumerate(soup.select('.book_cover')):
    d = {
        '이름': soup.select('.book_name')[i].text,
        '이미지': 'https://paullab.co.kr/bookservice/' + img['src'],
        '가격': soup.select('.book_info')[i*3].text\
            .replace('가격: ', '')\
            .replace('원', '')\
            .replace(',', '')\
            .replace('무료', '0')
    }
    result.append(d)

book_info_string = ''

for i in result:
    book_info_string += f'''
    <section>
        <img width="200px" src="{i['이미지']}">
        <h2>{i['이름']}</h2>
        <p>{i['가격']}</p>
    </section>
    '''

f = open('index.html', 'w')
s = f'''<html>
<head>
</head>
<body>
<h1>위니브 책 출판</h1>
{book_info_string}
</body>
</html>
'''
f.write(s)
f.close()
```

```py
# 개선안
# 개선된 코드

import requests
from bs4 import BeautifulSoup

data = requests.get('https://paullab.co.kr/bookservice/')
soup = BeautifulSoup(data.text, 'html.parser')

class Book:
    def __init__(self, name, image, price):
        self.name = name
        self.image = f'https://paullab.co.kr/bookservice/{image}'
        self.price = price

    def __str__(self):
        return f'<{self.name}, {self.price}>'

    def __repr__(self):
        return f'<{self.name}, {self.price}>'

    def make_html(self):
        return f'''
        <section>
            <img width="200px" src="{self.image}">
            <h2>{self.name}</h2>
            <p>{self.price}</p>
        </section>
        '''

    def make_json(self):
        return {
            '이름': self.name,
            '이미지': self.image,
            '가격': self.price
        }

result = []

for i, img in enumerate(soup.select('.book_cover')):
    name = soup.select('.book_name')[i].text
    img = img['src']
    price = soup.select('.book_info')[i*3].text\
            .replace('가격: ', '')\
            .replace('원', '')\
            .replace(',', '')\
            .replace('무료', '0')
    result.append(Book(name, img, price))

book_info_string = ''

for i in result:
    book_info_string += i.make_html()

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(book_info_string)
```

```py
import time
 
def job(number):
    print(f"Job {number} started")
    time.sleep(3)  # 이 time.sleep이 매우 오래 걸리는 작업 이라 가정하고 그 효율을 생각해봅시다. 일반 sleep은 CPU를 쉬게 합니다.
    print(f"Job {number} completed")
 
job(1)
job(2)
job(3)
```

```py
import nest_asyncio
nest_asyncio.apply()
import asyncio
 
async def job(number):
    print(f"Job {number} started")
    await asyncio.sleep(1) # 매우 오래 걸리는 작업, asyncio.sleep은 비동기 처리를 할 수 있도록 합니다.(다른 작업이 가능합니다.)
    print(f"Job {number} completed")
 
async def main():
    await asyncio.gather(job(1), job(2), job(3)) # await asyncio.wait([job(1), job(2), job(3)])
 
asyncio.run(main())
print('hello world')
```

```py
import asyncio
import random

async def generate_number():
    return ''.join(random.sample('123456789', 3))

async def get_user_guess():
    return await asyncio.to_thread(input, "3자리 숫자를 입력하세요: ")

async def check_guess(secret, guess):
    strikes = 0
    balls = 0
    for i in range(3):
        if guess[i] == secret[i]:
            strikes += 1
        elif guess[i] in secret:
            balls += 1
    return strikes, balls

async def game_round(secret):
    guess = await get_user_guess()
    strikes, balls = await check_guess(secret, guess)
    print(f"{strikes} 스트라이크, {balls} 볼")
    return strikes == 3

async def main():
    print("숫자 야구 게임을 시작합니다!")
    secret = await generate_number()
    rounds = 0
    while True:
        rounds += 1
        is_correct = await game_round(secret)
        if is_correct:
            print(f"정답입니다! {rounds}번 만에 맞추셨습니다.")
            break

    play_again = await asyncio.to_thread(input, "다시 플레이하시겠습니까? (y/n): ")
    if play_again.lower() == 'y':
        await main()
    else:
        print("게임을 종료합니다. 감사합니다!")

if __name__ == "__main__":
    asyncio.run(main())
```

### 정규표현식

1. 시작 ^010-[0-9]{4}-[0-9]{4}$ 끝
2. 범위 중간 부 무식 모든 h...o 구조 잡힘
3. [^] 아닌것 의미함 h[^ao]llo

```py
import re

list(s)
s2 = re.sub(r'[a-zA-Z]', lambda m: m.group() if m.group() in 'rev' else '', s)
print(s2)
s3 = ''.join(filter(lambda x: x in 'rev' or not x.isalpha(), s))
for match in re.finditer(pattern, s):
        num = int(match.group(1))
        if 1 <= num < 10:
            total += num
    
    return total
```

```py
import re

def solution(data):
    pattern = r'[rev](\d+)'
    total = 0
    for match in re.finditer(pattern, data):
        num = int(match.group(1))F
        if 1 <= num <= 10:
            total += num
        else:
            total += int(str(num)[0])
    s2 = str(total)
    return f'{s2[0]}월 {s2[1]}일'

data = 'a10b9r01ce33uab8wc1018v10cv111v9'
result = solution(data)
print(f"{result}")
```

```py
def solution(data):
    import re
    pattern = r'[rev](\d+)'
    total = 0
    for i in re.finditer(pattern, data):
        num = int(i.group(1))
        if 1 <= num <= 10:
            total += num
        else:
            total += int(str(num)[0])
    s2 = str(total)  
    return f'{s2[0]}월 {s2[1]}일'
F```
