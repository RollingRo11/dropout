from model import DropoutModel


def main():
    model = DropoutModel("names.txt")
    print("training...")
    model.train()

    print("\nmaking words:")
    names = model.generate_names()
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")


if __name__ == "__main__":
    main()
