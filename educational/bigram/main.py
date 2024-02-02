from model import BigramNameGenerator
from utils import illustrate_forward_process, illustrate_backward_process
from pathlib import Path

def main():
    # Initialize the generator with a sample dataset
    file_folder = Path(__file__).parent
    generator = BigramNameGenerator(file_folder.parent / "names.txt")

    # Generating plots
    generator.save_plot(None, "top_bigrams.pdf", "bigram")
    print("Saved top bigrams plot.")
    generator.save_plot(generator.N, "bigram_heatmap.pdf", "heatmap")
    print("Saved bigram heatmap plot.")

    # Test calculating the negative log likelihood of a specific string
    test_string = "andrej"
    res = generator.calculate_neg_log_likelihood(test_string)
    print(f"Neg Log Likelihood of '{test_string}': {res:.4f}")

    # Illustrate the forward process with one layer neural network
    word = "emma"
    illustrate_forward_process(generator, word)

    # Illustrate the backward process with one layer neural network
    word = "emma"
    illustrate_backward_process(generator, word)

    # Train with gradient descent
    generator.train_model(learning_rate=50, epochs=100)

    # Test name generation using the tabular bigram model
    print("Generated Names from the tabular bigram model:")
    generated_names = generator.generate_names_tabular(5)
    for name in generated_names:
        print(name)

    # Test name generation using the trained neural network model
    print("\nGenerated Names from the neural network model:")
    generated_nn_names = generator.generate_names_neural_network(5)
    for name in generated_nn_names:
        print(name)


if __name__ == "__main__":
    main()
