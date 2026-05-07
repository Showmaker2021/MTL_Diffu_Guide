import pandas as pd
import matplotlib.pyplot as plt


def plot_score(csv_path, output_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Extract epochs and scores
    epochs = data['epoch']
    scores = data['score']

    # Filter data to plot every 50 epochs
    filtered_epochs = epochs[epochs % 50 == 0]
    filtered_scores = scores[epochs % 50 == 0]

    # Plot the scores
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_epochs, filtered_scores, marker='o',
             linestyle='-', color='b', label='Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score over Epochs')
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Define the input CSV file and output plot file
    csv_file = "experiment/samples/train_log.csv"  # Adjust path if necessary
    output_file = "experiment/results/score_plot.png"

    # Call the plot function
    plot_score(csv_file, output_file)
    print(f"Plot saved as {output_file}")
