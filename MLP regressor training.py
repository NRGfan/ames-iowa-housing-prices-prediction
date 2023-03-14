import torch
import torch.nn as nn
import pandas as pd

loss_trace = []


# defining the model architecture
class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)


# Dataset gen
class housing_dataset():
    def __init__(self, csv_path):
        csv_data = pd.read_csv(csv_path)
        self.csv_data = pd.get_dummies(csv_data)
        self.num_samples = len(self.csv_data)
        input_param_list = ["Id", "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "1stFlrSF",
                            "2ndFlrSF",
                            "YearRemodAdd", "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr",
                            "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "YrSold", "MoSold"]
        global num_inputs
        num_inputs = len(input_param_list)
        self.params = []
        for i in range(self.num_samples):
            house_param = list(csv_data.iloc[i][input_param_list])
            house_param = torch.tensor(house_param)
            self.params.append(house_param)

    def __getitem__(self, index):
        price = torch.tensor(self.csv_data.iloc[index]["SalePrice"])
        return self.params[index], price

    def __len__(self):
        return self.num_samples


class housing_dataset_test():
    def __init__(self, csv_path):
        csv_data = pd.read_csv(csv_path)
        self.csv_data = pd.get_dummies(csv_data)
        self.num_samples = len(self.csv_data)
        input_param_list = ["Id", "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "1stFlrSF",
                            "2ndFlrSF",
                            "YearRemodAdd", "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr",
                            "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "YrSold", "MoSold"]
        global num_inputs
        num_inputs = len(input_param_list)
        self.params = []
        for i in range(self.num_samples):
            house_param = list(csv_data.iloc[i][input_param_list])
            house_param = torch.tensor(house_param)
            self.params.append(house_param)

    def __getitem__(self, index):
        return self.params[index]

    def __len__(self):
        return self.num_samples


train_csv_path = r"C:\Users\Joshua\Downloads\kaggle_housingpred\train.csv"
train_set = housing_dataset(train_csv_path)
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=8)

mlp = MLP()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-5)

 #Run the training loop
if __name__ == "__main__":
    mlp = torch.load(r"C:\Users\Joshua\Downloads\kaggle_housingpred\MLP_baseline.pth")
    for epoch in range(0, 30):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
        print("Epoch %s loss: %s"%(epoch+1, current_loss))
        loss_trace.append(current_loss)
    print('Training process has finished.')
    print(loss_trace)
    path = r"C:\Users\Joshua\Downloads\kaggle_housingpred\MLP_baseline.pth"
    torch.save(mlp, path)



