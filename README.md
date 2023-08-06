# is this a bible?
 An IA model that detects whether a given verse is from the Bible or not

The model presents capabilities at a very high recognition level, for the Hebrew language.
The complete dataset on which the model was trained is stored in the `bible_data.csv` file.

You can try the model's capabilities easily,By downloading the release file from here - https://github.com/NHLOCAL/is-this-bible/releases/download/v1.0/is-this-bible.zip.

**To run the model, download the following libraries using pip**:

`nltk`, `joblib`.

-----

**דוגמה:**

קלט שלילי:
```shell
try_model.py "בגיטהאב ניתן להעלות מערכות קוד פתוח"
```
פלט:


```shell
Text: בגיטהאב ניתן להעלות מערכות קוד פתוח | Prediction: Other | Confidence Score: 0.0340
```
קלט חיובי:

```shell
try_model.py "עניה סערה לא נחמה הנה אנכי מרביץ בפוך אבניך"
```
פלט:

```shell
Text: עניה סערה לא נחמה הנה אנכי מרביץ בפוך אבניך ויסדתיך בספירים | Prediction: Bible | Confidence Score: 1.0000
```
