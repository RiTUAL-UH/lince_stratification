# LinCE Stratification

This repo provides a demo of the stratification strategy proposed in the [LinCE paper](https://www.aclweb.org/anthology/2020.lrec-1.223.pdf). 
Note that the original data used to recreate the splits is omitted since we do not intend to distribute the test labels.
Thus, we provide a demo notebook of the method assuming that our entire corpus is a tranining set from one of the datasets (i.e., Hin-Eng NER).

Feel free to get in touch with us if you have any questions or comments using Github issues or contacting us through the emails in the paper.

If you find this code useful in your projects, please cite our paper:
```
@inproceedings{aguilar-etal-2020-lince,
    title = "{L}in{CE}: A Centralized Benchmark for Linguistic Code-switching Evaluation",
    author = "Aguilar, Gustavo  and
      Kar, Sudipta  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.223",
    pages = "1803--1813",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
```