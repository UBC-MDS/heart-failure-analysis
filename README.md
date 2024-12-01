# Heart Failure Analysis

-   contributors: Yuhan Fan, Gurmehak Kaur, Ke Gao, Merari Santana

## About

In this project, we attempt to build a classification model using logistic regression algorithm to predict patient mortality risk after surviving a heart attack using their medical records. Using patient test results, the final classifier achieves an accuracy of 81.6%. The model’s precision of 70.0% suggests it is moderately conservative in predicting the positive class (death), minimizing false alarms.More importantly, the recall of 73.68% ensures the model identifies the majority of high-risk patients, reducing the likelihood of missing true positive cases, however, there is still room for a lot of improvement, particularly in aiming to maximise recall by minimising False Negatives. The F1-score of 0.71 reflects a good balance between precision and recall, highlighting the model’s robustness in survival prediction. While promising, further refinements are essential for more reliable predictions and effectively early intervention.

The data set used in this project was created by D. Chicco, Giuseppe Jurman in 2020. It was sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records). Each row in the data set represents the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features（age, anaemia, diabetes, platelets, etc.).

## Dependencies

- Docker
- VS Code
- VS Code Jupyter Extension

## Usage

### Setup

> If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub repository.


### Running the analysis

1. Navigate to the root of this project on your computer using the
   command line and enter the following command:

``` 
docker compose up
```

2. In the terminal, look for a URL that starts with 
`http://127.0.0.1:8888/lab?token=` 
 
Copy and paste that URL into your browser.

3. To run the analysis,
open `heart-failure-analysis.ipynb` in Jupyter Lab you just launched
and under the "Kernel" menu click "Restart Kernel and Run All Cells...".

### Clean up

1. To shut down the container and clean up the resources, 
type `Cntrl` + `C` in the terminal
where you launched the container, and then type `docker compose rm`

## Developer notes

### Developer dependencies
- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)

### Adding a new dependency

1. Add the dependency to the `environment.yml` file on a new branch.

2. Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the `conda-linux-64.lock` file.

2. Re-build the Docker image locally to ensure it builds and runs properly.

3. Push the changes to GitHub. A new Docker
   image will be built and pushed to Docker Hub automatically.
   It will be tagged with the SHA for the commit that changed the file.

4. Update the `docker-compose.yml` file on your branch to use the new
   container image (make sure to update the tag specifically).

5. Send a pull request to merge the changes into the `main` branch. 

## License

This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/legalcode).

If re-using/re-mixing please provide attribution and link to this webpage. The software code contained within this repository is licensed under the MIT license. See [the license file](LICENSE.md) for more information.

## References

Chicco, D., Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak 20, 16 (2020). <https://doi.org/10.1186/s12911-020-1023-5>

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. <http://archive.ics.uci.edu/ml>.

Heart Failure Clinical Records [Dataset]. (2020). UCI Machine Learning Repository. <https://doi.org/10.24432/C5Z89R>.

Kuter, David J. "Thrombopoietin and Platelets: Count Regulation." American Journal of Hematology. Wiley Online Library, 2016. Accessed November 27, 2024. https://onlinelibrary.wiley.com/doi/full/10.1002/ajh.23704

Mayo Clinic Staff. "Ejection Fraction: What Does It Measure?" Mayo Clinic, Mayo Foundation for Medical Education and Research. Accessed November 27, 2024. https://www.mayoclinic.org/tests-procedures/ekg/expert-answers/ejection-fraction/faq-20058286

Medical News Today Staff. "What to Know About Serum Creatinine." Medical News Today. Last updated March 22, 2018. Accessed November 27, 2024. https://www.medicalnewstoday.com/articles/322380

MedlinePlus. "Serum Sodium Test." U.S. National Library of Medicine. Last reviewed September 5, 2023. Accessed November 27, 2024. https://medlineplus.gov/ency/article/003481.html

U.S. News Staff. "Creatinine Levels: What You Should Know." U.S. News & World Report. Accessed November 27, 2024. https://health.usnews.com/health-care/patient-advice/articles/creatininelevels
