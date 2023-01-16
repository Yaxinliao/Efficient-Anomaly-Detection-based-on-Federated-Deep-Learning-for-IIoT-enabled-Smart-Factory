import codes.algos as al
import codes.utils as ut

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = ut.load_data("../input/test_3+6项独热编码.csv",
                                                    "../input/train_3+6项独热编码.csv")
    al.dt(x_train, y_train, x_test, y_test)
    # al.fcm(x_train, y_train, x_test, y_test)
