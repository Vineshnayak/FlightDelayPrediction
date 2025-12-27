## **🚀 Flight Delay Prediction & Visualization**

**Machine learning project that predicts flight arrival delays and visualizes delay behaviour across airports and time periods.**
The script loads `flight_delay_predict.csv`, performs **EDA and visualizations**, scales selected features, trains a **TensorFlow neural network**, evaluates the model, and supports **real-time user input prediction** from the console. 

### **Tech Stack**

**Python · Pandas · NumPy · Plotly · Matplotlib · Seaborn · Scikit-learn · TensorFlow**

### **Key Features**

* Converts flight date to datetime
* Visualizes average delay by **origin**, **destination**, and **month**
* Generates a **correlation heatmap**
* Trains a neural network using
  **AirTime + Distance → Delay minutes & delay flag**
* Supports **user-input prediction**
* Saves the trained model

### **Install Requirements**

Dependencies are listed in `requirements.txt`:
pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, **tensorflow==2.15**, **protobuf==3.20.*** 

```bash
pip install -r requirements.txt
```

### **Run**

```bash
python Complete_code.py
```

Place `flight_delay_predict.csv` in the same directory.

