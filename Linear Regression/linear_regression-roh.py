import tensorflow as tf
import numpy as num

def test_random_seed(seed=777):
    '''set_random_seed 는 Graph 레벨에서의 Random 인자를 셋팅하게 한다는 의미로써
    이런 식으로 만들어 놓고, 각 Operation (like random_normal) 에서 별다른 seed 에 대해
    명시하지 않을 경우에 항상 동일한 확률인자(?)가 사용된다고 보면된다. 즉 Graph 단위에서
    인자를 이렇게 만들어 놓으면 매번 프로그램을 만들 때 마다 아래와 같이 고정된 값이 나오게 된다.
    Random Normal 1:  [ 2.20866942] [ 0.95267582] [ 1.01215756]
    Random Normal 2:  [ 1.791412] [ 0.26029482] [ 0.77326351]
    Random Normal 3:  [-0.44488874] [ 0.803523] [ 1.37893617]
    '''
    tf.set_random_seed(seed)

def verify_set_random_seed_function():
    test_random_seed()

    #random_normal의 경우는 random하게 하나의 값을 뽑아온다고 생각하면 될까?
    #normal disrtribution에서 하나의 값을 가져온다고 보면 될 듯 하며, normal distribution은
    #일종의 가우시안 분포 그래프처럼 생각하면 될 듯...
    randA = tf.random_normal([1])
    randB = tf.random_normal([1])
    randC = tf.random_normal([1])

    print("What is randA : ", randA)
    print("What is randB : ", randB)
    print("What is randB : ", randC)

    with tf.Session() as sess1:
        print("Random Normal 1: ", sess1.run(randA), sess1.run(randB), sess1.run(randC))
        print("Random Normal 2: ", sess1.run(randA), sess1.run(randB), sess1.run(randC))
        print("Random Normal 3: ", sess1.run(randA), sess1.run(randB), sess1.run(randC))

    with tf.Session() as sess2:
        print("Random Normal 1: ", sess2.run(randA), sess2.run(randB), sess2.run(randC))
        print("Random Normal 2: ", sess2.run(randA), sess2.run(randB), sess2.run(randC))
        print("Random Normal 3: ", sess2.run(randA), sess2.run(randB), sess2.run(randC))

def inversing_data(array):
    return [x for x in array[::-1]]


if __name__ == '__main__':
    test_random_seed()

    x_train = [1, 2, 3]
    y_train = [1, 2, 3]

    '''
    Our Hypotesis (W*X + b = Y) first weight and bias.
    '''
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = x_train * W + b
    costfunc = tf.square(hypothesis - y_train)

    '''
    Linear Regression의 경우 x_train, y_train 으로 명명된 2차원 graph 상에 포진된 각 점들과의 
    거리가 가장 가까운 선을 찾는 것이고, 이 선은 W, b 값을 변경해 가면서 찾아야 하는데 이를 
    자동적으로 찾기 위해 reduce_mean 이라는 방식을 사용한다.
    '''

    cost = tf.reduce_mean(costfunc); #Aggregate all elements in hypothesis and divide it by hypothesis size

    #Gradient Descent 방식으로 cost가 가장 작아지는 방향으로 hypothesis의 W, b 값을 변경해 가며 계산을 한다.
    #여기서 계산되어지는 W, b는 learning_rate에 따라 변화되는 폭(step)이 달라진다.
    #이미 알겠지만 W, b에 따라서 xy 평면에 그려지는 line의 기울기가 변한다. 즉 learning_rate는 이 기울기가
    #변화되는 폭을 정한다고 볼 수 있다.

    # Gradient Descent Optimizer는 W = W - tf.mul(0.001, cost(constfunc*X)) 를 내부에서 수행한다.

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cost)

    sess1 = tf.Session()
    sess1.run(tf.global_variables_initializer())
    with sess1:
        for step in range(10001):
            sess1.run(train)
            if step % 20 == 0:
                print(step, sess1.run(cost), sess1.run(W), sess1.run(b))


