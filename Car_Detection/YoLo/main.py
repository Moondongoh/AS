from model_training.train_model import train_model
from model_training.present_a import config as present
from model_training.past_b import config as past

if __name__ == "__main__":
    print("Training on Dataset Present")
    train_model(present)
    
    print("Training on Dataset Past")
    train_model(past)


# from model_training.train_model_balance import train_model
# from model_training.present_a import config as present
# from model_training.past_b import config as past

# if __name__ == "__main__":
#     # print("Training on Dataset Present")
#     # train_model(present)
    
#     print("Training on Dataset Past")
#     train_model(past)
