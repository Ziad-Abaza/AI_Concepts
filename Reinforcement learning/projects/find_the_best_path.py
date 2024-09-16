
############################################################
# أولا نقوم بتهيئة البيئة
############################################################

grid = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 10, 0],
]

# الحالات (State) تمثل المواقع
states = [(0, 0),(0, 1),(0, 2),
          (1, 0),(1, 1),(1, 2),
          (2, 0),(2, 1),(2, 2),]

# الإجراءات الممكنة
actions = ["up", "down", "left", "right"]

# تحديد الحدود
def is_valid_move(state):
    x, y = state
    return 0 <= x < 3 and 0 <= y < 3

def move(state, action):
    x, y = state
    if action == "up":
        new_state = (x - 1, y)
    elif action == "down":
        new_state = (x + 1, y)
    elif action == "left":
        new_state = (x, y - 1)
    elif action == "right":
        new_state = (x, y + 1)
    
    # التأكد من أن الحركة صالحة
    if is_valid_move(new_state):
        return new_state
    return state  # إذا كانت الحركة غير صالحة يبقى في نفس المكان

############################################################
# ثانيا نقوم بتهيئة الوكيل
############################################################

import random
# يتم تحديث القيم في Table مع مرور الوقت باستخدام خوارزمية Q-Learning 
# بناءً على التجارب التي يخوضها الوكيل.
q_table = {
    (0, 0): {action: 0 for action in actions},
    (0, 1): {action: 0 for action in actions},
    (0, 2): {action: 0 for action in actions},
    (1, 0): {action: 0 for action in actions},
    (1, 1): {action: 0 for action in actions},
    (1, 2): {action: 0 for action in actions},
    (2, 0): {action: 0 for action in actions},
    (2, 1): {action: 0 for action in actions},
    (2, 2): {action: 0 for action in actions},    
}


# المعلمات
alpha = 0.1  # معدل التعلم
# قيمة alpha بين 0 و 1:
# إذا كانت alpha قريبة من 1، فإن الوكيل يعتمد بشكل كبير على المعلومات الجديدة،
#ويتعلم بسرعة (لكن قد يؤدي ذلك إلى تقلبات في القيم).
# إذا كانت alpha قريبة من 0، فإن الوكيل يعتمد بشكل كبير على الخبرات القديمة ويتعلم ببطء.

gamma = 0.9  # معامل الخصم
# قيمة gamma بين 0 و 1:
# إذا كانت gamma قريبة من 1، فإن الوكيل يولي اهتمامًا كبيرًا بالمكافآت المستقبلية.
# إذا كانت gamma قريبة من 0، فإن الوكيل يهتم أكثر بالمكافآت الفورية.

epsilon = 0.6  # معدل الاستكشاف
# قيمة epsilon بين 0 و 1:
# إذا كانت epsilon قريبة من 1، فإن الوكيل يستكشف البيئة أكثر (يتخذ إجراءات عشوائية في معظم الأحيان).
# إذا كانت epsilon قريبة من 0، فإن الوكيل يستغل ما تعلمه أكثر 
# (يتخذ الإجراءات التي يعتبرها أفضل استنادًا إلى القيم الحالية في جدول Q).

# اختيار إجراء باستخدام سياسة epsilon-greedy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  
    return max(q_table[state], key=q_table[state].get)  

# تحديث جدول Q
def update_q_table(state, action, reward, next_state):
    best_next_action = max(q_table[next_state], key=q_table[next_state].get)
    q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])


############################################################
# وتشغيل الخوارزمية
############################################################

# تجارب متعددة لتعلم الوكيل
for episode in range(1000):
    state = (0, 0)  # البداية دائماً من [0, 0]
    
    while state != (2, 1):  # تكرار حتى يصل الوكيل إلى الهدف
        action = choose_action(state)
        next_state = move(state, action)
        
        # تحديد المكافأة
        reward = grid[next_state[0]][next_state[1]] if next_state == (2, 1) else -1
        
        # تحديث جدول Q
        update_q_table(state, action, reward, next_state)
        
        state = next_state  # الانتقال إلى الحالة الجديدة

print("The best  action for each state is:")
print("Q-Table after learning:")
for state, actions in q_table.items():
    print(f"State {state}: {actions}")
