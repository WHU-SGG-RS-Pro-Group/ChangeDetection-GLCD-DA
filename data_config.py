
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'ChangeDetection':
            self.root_dir = 'California'  #训练集的根目录自己给出
        elif data_name == 'predict':
            self.root_dir = './SAMPLES_CALIFORNIA'# 用来验证效果的测试数据放在了quik_start控制的路径下
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='ChangeDetection')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

