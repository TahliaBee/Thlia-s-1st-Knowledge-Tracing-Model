import torch.optim
from torch.utils.tensorboard import SummaryWriter
from model import *
from dataset import *
import time
from eval import *
import os


class Train_Processor:
    def __init__(self, model, train_loader, test_loader, loss_fn, optim, total_epoch, threshold, model_path='../models', writer_path='../logs'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optim = optim
        self.total_epoch = total_epoch
        self.threshold = threshold
        self.writer = SummaryWriter(writer_path) if writer_path is not None else None
        self.epoch = 0
        self.total_train_step = 0
        self.total_test_step = 0
        self.model_path = model_path
        self.best_accuracy = 0
        self.best_loss = float('inf')

    def train(self):
        start_time = time.time()
        for epoch in range(total_epoch):
            self._train_one_epoch()
        end_time = time.time()
        print(f'DONE, time used: {end_time - start_time:.4f}s')


    def _train_one_epoch(self):
        # a whole process in a epoch, including a training process and a testing process
        # for batch: calc loss, backward(is_test=False), record accuracy, record in writer, total_train_step update
        # return accuracy
        model = self.model
        for epoch in range(self.total_epoch):
            # train
            model.train()
            time0 = time.time()
            train_loss, train_accuracy = self._train_or_test_component(is_test=False)
            time1 = time.time()
            print(f'epoch train time: {time1 - time0:.4f}s')

            # eval
            model.eval()
            time0 = time.time()
            test_loss, test_accuracy = self._train_or_test_component(is_test=True)
            time1 = time.time()
            print(f'epoch test time: {time1 - time0:.4f}s')

            # print epoch info
            print("epoch: {}, train_loss: {}, test_loss:{}, train accuracy: {}, test accuracy: {}"
                  .format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

            # save regularly
            if epoch % 10 == 0:
                time0 = time.time()
                self._save_model(str(epoch // 10)+'.pth', f'epoch {str(epoch // 10)}')
                time1 = time.time()
                print(f'model saving time: {time1 - time0:.4f}s')

            # update best info
            if test_accuracy > self.best_accuracy:
                self._save_model('best_accuracy_model.pth', 'best accuracy')
                self.best_accuracy = test_accuracy
            if test_loss < self.best_loss:
                self._save_model('best_loss_model.pth', 'best loss')
                self.best_loss = test_loss


    def _train_or_test_component(self, is_test=False):
        working_dataset = self.test_loader if is_test else self.train_loader
        title = 'Testing' if is_test else 'Training'
        loss_fn = self.loss_fn
        writer = self.writer
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        total_working_step = self.total_test_step if is_test else self.total_train_step
        for working_data, response in tqdm(working_dataset, desc=f'{title} - Epoch {self.epoch}'):
            response = torch.squeeze(response, dim=1)
            # only no grad when test
            if is_test:
                with torch.no_grad():
                    predict = model(working_data, response)
            else:
                predict = model(working_data, response)

            response = torch.flatten(response).float()
            predict = torch.flatten(predict)

            # flatten
            mask = response.flatten() > -0.9  # 有效样本掩码
            valid_response = response.flatten()[mask].float()
            valid_predict = predict.flatten()[mask]

            # calculate loss
            loss = loss_fn(valid_predict, valid_response)
            total_loss += loss.item()

            # backward
            if not is_test:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            # record accuracy
            correct = ((valid_predict >= self.threshold) == (valid_response >= self.threshold)).float().sum()
            total_correct += correct.item()
            total_predictions += valid_response.numel()

            # writer record
            if total_working_step % 100 == 0:
                text = 'test' if is_test else 'train'
                if writer is not None:
                    writer.add_scalar(f'{text}_loss', loss.item(), total_working_step)

            # increment
            if is_test:
                self.total_test_step += 1
            else:
                self.total_train_step += 1
        # calculate accuracy
        accuracy = total_correct / total_predictions
        return total_loss, accuracy

    def _save_model(self, model_name, text):
        path = os.path.join(self.model_path, model_name)
        torch.save(self.model, path)
        print(f'{text} model saved')


if __name__ == '__main__':
    start_time = time.time()

    num_processes = 10 # multiprocessing
    # num_processes = 1

    batch_size = 32
    dataset_name = 'assist09'
    n_problem, n_skill, n_qtype = 17751, 149, 5
    d_model = 256
    learning_rate = 0.001
    total_epoch = 300
    threshold = 0.5
    dummy = False

    train_dataset = KTData('train',
                           dataset_name,
                           dummy=dummy)
    test_dataset = KTData('test',
                          dataset_name,
                          dummy=dummy)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_processes  # multiprocessing
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_processes  # multiprocessing
    )

    model = MyModel(n_problem, n_skill, n_qtype, d_model)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    init_time = time.time()
    print("initiating spent:{:.4f}s".format(init_time - start_time))

    tp = Train_Processor(model,
                         train_loader,
                         test_loader,
                         loss_fn,
                         optim,
                         total_epoch,
                         threshold)
    tp.train()


    # model.share_memory()  # multiprocessing
    # processes = []
    # for rank in range(num_processes):
    #     p = mp.Process(target=train, args=(model, train_loader, test_loader, loss_fn, optim, total_epoch, threshold))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
