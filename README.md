# GPU_tests

# Что попробовать сделать для исправления

1. Вывести act_values в функции DQNAgent. Раз нам выводятся NaN надо тут посмотреть
    def predict(self, state):
        act_values = self.model.predict(state) #ценность действий| влево - ценность 0.5, вправо - ценность 0.3. Поэтому выбираем лево
        return act_values
