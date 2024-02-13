# GPU_tests

# Что попробовать сделать для исправления

1. Вывести act_values в функции DQNAgent. Раз нам выводятся NaN надо тут посмотреть


   def predict(self, state):
        act_values = self.model.predict(state) #ценность действий| влево - ценность 0.5, вправо - ценность 0.3. Поэтому выбираем лево
        return act_values


2. Мы в него подаём пустой (np.zeros) state. Может ошибка поэтому? Но почему при малых количествах ошибка не возникает?
3. Перепроверить state_size в agend.load()
