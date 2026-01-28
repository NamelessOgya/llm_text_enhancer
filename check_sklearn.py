try:
    import sklearn
    print("sklearn installed")
    from sklearn.feature_extraction.text import CountVectorizer
    print("CountVectorizer importable")
except ImportError as e:
    print(f"ImportError: {e}")
