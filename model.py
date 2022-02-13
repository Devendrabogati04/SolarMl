import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6
from statsmodels.tsa.arima_model import ARIMA
import xlrd

def service_model(uploaded_files):
    df = pd.read_excel(uploaded_files,sheet_name='Service')
    # print(df.head())


    df = df.select_dtypes(exclude=['datetime64'])
    df = df.fillna(df.mean())
    x = df.drop('Service Required?', axis=1)
    y = df['Service Required?']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

    # st_x = StandardScaler()
    # x_train = st_x.fit_transform(x_train)
    # x_test = st_x.transform(x_test)

    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    return y_pred, x_test, df

def roi_pred(uploaded_files):
    df = pd.read_excel(uploaded_files,sheet_name='Roi Data')
    x = df.drop(['Payback Period(yr)', ' ROI ($)'], axis=1)
    y = df[['Payback Period(yr)', ' ROI ($)']]

    model = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.005, random_state=1)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return y_pred,x_test,df


def efficiency_pred(uploaded_files):
    dup=pd.read_excel(uploaded_files,sheet_name='Performance')

    df= pd.read_excel(uploaded_files,sheet_name='Performance')
    df.drop(df.columns[[0, 1, 2, 4, 5, 6, 7]], axis=1, inplace=True)
    indexed_df = df.set_index(['Month'])

    # rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(indexed_df, model='additive')
    # fig = decomposition.plot()
    # plt.show()

    model = ARIMA(indexed_df, order=(2, 1, 2))
    results_ARIMA = model.fit(disp=-1)

   

    return dup, decomposition,results_ARIMA


def solar_generation(uploaded_files):
    df = pd.read_excel(uploaded_files,sheet_name='Generation')
    df_dup = pd.read_excel(uploaded_files,sheet_name='Generation')
    df['Hour'] = df['First Hour of Period']
    df['TimeStamp'] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day, hour=df.Hour))

    x = df[['Year', 'Month', 'Day', 'Hour', 'Is Daylight', 'Average Temperature (Day)', 'Average Wind Direction (Day)',
            'Sky Cover', 'Visibility', 'Relative Humidity']]
    y = df['Power Generated']

    model = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.002, random_state=1)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return y_pred,x_test,df_dup,df

