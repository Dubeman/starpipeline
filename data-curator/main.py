import os
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from DataCurator import DataCurator
from parameters import dataset_path, output_path
from DAE import DAE
import matplotlib.pyplot as plt

def main():
    
    # Initialize and run the pipeline
    pipeline = DataCurator(dataset_path, output_path)
    
    logging.info("Starting data curation pipeline...")
    
    # Load the data
    pipeline.load_data()
    
    # Split the data into different noise groups
    train_loader, val_loader = pipeline.split_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAE().to(device)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        print("Model loaded from checkpoint.")
    else:
        print("No checkpoint found, starting training from scratch.")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        num_epochs = 50
        patience = 5
        best_val_loss = float('inf')
        epochs_no_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            for noisys, clean in train_loader:
                for noisy in noisys:
                    noisy = noisy.to(device)
                    clean = clean.to(device)

                    optimizer.zero_grad()
                    outputs = model(noisy)
                    loss = criterion(outputs, clean)
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for noisys, clean in val_loader:
                    for noisy in noisys:
                        noisy, clean = noisy.to(device), clean.to(device)
                        outputs = model(noisy)
                        val_loss = criterion(outputs, clean)
                        total_val_loss += val_loss.item()
        
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {total_val_loss / len(val_loader):.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print("Early stopping triggered.")
                break
            torch.save(model.state_dict(), "model.pth")
    model.eval()
        
    noisy_imgs, clean_imgs = next(iter(train_loader))

    noisy_img = noisy_imgs[1]
    clean_img = clean_imgs[1]
    noisy_img = noisy_img.permute(1, 2, 0).cpu().numpy()
    clean_img = clean_img.permute(1, 2, 0).cpu().numpy()

    noisy_img = np.clip(noisy_img, 0, 1)
    clean_img = np.clip(clean_img, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(noisy_img)
    axes[0].set_title("Noisy Image")
    axes[0].axis('off')

    axes[1].imshow(clean_img)
    axes[1].set_title("Clean Image")
    axes[1].axis('off')

    plt.show()

    logging.info("Data curation pipeline completed successfully!")

if __name__ == "__main__":
    main()
