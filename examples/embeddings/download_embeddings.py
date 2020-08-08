import chakin


if __name__ == '__main__':
    chakin.search(lang='Japanese')
    """
                            Name  Dimension     Corpus VocabularySize              Method  Language                 Author
    6                fastText(ja)        300  Wikipedia           580K            fastText  Japanese               Facebook
    22  word2vec.Wiki-NEologd.50d         50  Wikipedia           335K  word2vec + NEologd  Japanese  Shiroyagi Corporation
    """
    chakin.download(number=22, save_dir='./')
