def test_pipeline():
    print("Running full pipeline test...")
    import preprocess
    import train
    import evaluate

    preprocess.main()
    train.train()
    evaluate.evaluate()

    print("Pipeline test completed successfully!")


if __name__ == "__main__":
    test_pipeline()
