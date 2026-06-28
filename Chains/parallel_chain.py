from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


model1 = ChatOllama(model = "tinyllama")

model2 = ChatOllama(model = 'gemma:latest')


prompt1 = PromptTemplate(
    template= 'Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']

)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single documents \n {notes} and {quiz}',
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 |model2 | parser

chain = parallel_chain | merge_chain

text = """Data-Driven Duration Management: Term Structure Forecasting
Using Machine Learning
Tobias LausserORCID Joao Eduardo VuoloORCID
Rudi ZagstORCID
Department of Mathematics, School of Computation, Information and Technology
Technical University of Munich, Parkring 11, Garching bei München, Bavaria, Germany
tobias.lausser@tum.de
Preprint.
Abstract
This paper compares different methods for forecasting the term structure of U.S. and
European zero-coupon government bonds using both traditional econometric and Machine
Learning (ML) approaches. We compare classical models (e.g., Dynamic Nelson-Siegel (DNS)
and Principal Component Analysis (PCA)) with different Neural Network (NN) architectures,
including those inspired by the classical models, on the U.S. Treasury market and bonds issued
by the European Central Bank (ECB). To enhance predictive performance, macroeconomic
variables are incorporated. The findings for both markets are separately analyzed and
compared. To this end, we propose a robust model evaluation framework combining statistical
accuracy metrics—such as RMSE, MAE, and directional accuracy—with the economic
relevance of a quantitative bond trading strategy. Results show that NNs consistently
outperform traditional models in both forecasting accuracy and portfolio performance. For
the U.S., the most effective approach is a direct-forecasting NN that incorporates DNS
factors to reduce the dimensionality of zero-rate data and an Autoencoder (AE) to extract
macroeconomic features, while for Europe, the optimal model is a factor-based NN using
PCA-derived zero-rate factors without the integration of macroeconomic variables. Overall,
the paper demonstrates how combining traditional modeling approaches with modern ML
techniques and evaluation can improve yield curve forecasts and support applications in
fixed-income portfolio construction.
Keywords: Term Structure, Machine Learning, Autoencoder, Duration Management, NelsonSiegel, PCA
1 Introduction
The bond market is a key pillar of the global financial system, reflecting market expectations
regarding interest rates, inflation, and economic growth. As of March 2025, U.S. gross national
debt is about $37 trillion, compared to a stock market capitalization of $49.8 trillion. In the EU,
government debt stood at roughly €13.3 trillion (81% of GDP) by end-2024, while stock market
capitalization is projected at $13.39 trillion in 2025.
1
arXiv:2606.26815v1 [q-fin.PM] 25 Jun 2026
The primary objective of our investigations is the term structure of interest rates, also known as
the yield curve. It represents the relationship between bond yields and time to maturity (Fabozzi,
2012). Institutional investors, including pension funds and insurance companies, typically hold
large proportions of fixed-income securities (Blake et al., 2001), making yield curve forecasting
both academically and practically significant. Moreover, interest rate changes directly impact
the valuation of all financial assets (Diebold and Li, 2006; Cochrane and Piazzesi, 2005).
One reason for the complexity of term structure forecasting is its high dimensionality (Cochrane
and Piazzesi, 2005; Diebold and Li, 2006). Prominent approaches to modeling the term structure
involve compressing its information to a few latent factors. A foundational example is the NelsonSiegel (NS) model from Nelson and Siegel (1987), later extended into a dynamic framework by
Diebold and Li (2006). This model parsimoniously describes the zero-rate curve using three
intuitive factors interpreted as level, slope, and curvature. The forecasting process is then
two-staged: first, the latent factors are extracted from the term structure at each point in time,
and second, the dynamics of those three factors are modeled with Autoregressive (AR) and
Vector Autoregressive (VAR) time-series models. Building on this, Christensen et al. (2011)
developed the Arbitrage-Free Nelson-Siegel (AFNS) model, which imposes no-arbitrage conditions
to ensure theoretical consistency, albeit at the cost of increased complexity requiring estimation
via state-space methods like the Kalman Filter (KF). In a parallel line of research, Ang and
Piazzesi (2003) demonstrated the importance of macroeconomic variables, incorporating them
alongside latent factors within a no-arbitrage VAR framework to model the zero-rate curve.
More recently, Salachas et al. (2024) examined the predictive power of the zero-rate curve
during the COVID-19 pandemic by comparing traditional term structure models with versions
augmented by pandemic-related information. Their study applied DNS and AFNS, as well as
VAR models, on alternative latent factors. The authors incorporated COVID-19 indicators—such
as infection rates, mobility measures, and government stringency indices—into the models
to capture the macro-financial disruptions of this period. They provide evidence that these
enhanced-information models significantly improve short- and medium-term interest rate forecasts,
highlighting the predictive power of such exogenous variables.
These traditional models typically rely on linearity assumptions and may fall short in capturing
the nonlinear and dynamic nature of zero-rate curve movements, particularly during periods of
market stress, structural breaks, or unconventional monetary policy regimes. As financial markets
have become increasingly complex and volatile, there is growing interest in exploring more flexible
modeling techniques that can better account for nonlinearities and higher-order interactions
among variables, an area where Machine Learning (ML) methods have demonstrated promising
results (Vela, 2013; Nunes et al., 2019). The most prominent example of ML architecture are
Neural Networks (NN).
Early research into NNs for zero-rate curve modeling focused on forecasting the latent factors
of traditional parametric models. For instance, Vela (2013) compared different NN architectures
for predicting the factors of the NS model against AR and VAR. While finding that NNs performed
best on U.S. data, the authors noted inconsistent results across different markets, concluding
that there was insufficient evidence to declare a universally superior forecasting method.
Shifting the focus from factor models to direct zero-rate prediction, subsequent research
2
underscored the importance of incorporating macroeconomic data. In their work on the European
zero-rate curve, Nunes et al. (2019) provided a comprehensive comparison between two NN
architectures: Single-Task Learning (STL), where an independent model is trained for each
maturity, and Multi-Task Learning (MTL), which forecasts all maturities jointly using a shared
NN. Their results showed that NNs, particularly the MTL approach, consistently outperformed
traditional linear benchmarks, such as Ordinary Least Squares (OLS). A key contribution of their
work was the rigorous feature selection process, which began with a broad set of macroeconomic
variables and systematically narrowed it down to the most informative predictors. The final set
primarily included variables capturing real economic activity, such as the unemployment rate
and industrial production growth, as well as inflation dynamics, represented by inflation indices.
This process demonstrated that incorporating macroeconomic information of these two kinds
significantly enhances the forecasting accuracy of NN models.
Beyond statistical accuracy, the practical economic value of forecasting models is a crucial
consideration. Dunis and Morrison (2007) addressed this by comparing Autoregressive Moving
Average (ARMA), KF models, and NNs not only on statistical metrics like Root Mean Squared
Error (RMSE) but also on their performance in a directional trading strategy. Their findings
were significant: while the ARMA model achieved the lowest forecast error, the NN-based
strategy generated the highest risk-adjusted returns, as measured by the Sharpe Ratio. This
underscores that statistical superiority does not always translate to better economic outcomes,
making application-specific evaluation essential.
Autoencoders (AEs) have been explored for their ability to perform non-linear dimensionality
reduction on the zero-rate curve. Suimon et al. (2020) used an AE to decompose the Japanese
zero-rate curve into interpretable factors analogous to the classic level, slope, and curvature
of the DNS model. They then developed a trading strategy based on identifying mispricings,
where the AE-reconstructed rate deviates from the actual rate. Although forecasting models like
Long Short-Term Memory (LSTM) networks produced higher returns, the AE-based strategy
yielded stable positive returns and offered greater interpretability, showcasing its utility in bond
valuation and market analysis.
This study advances the literature on zero-rate curve forecasting by conducting a comprehensive and methodologically rigorous comparison of competing modeling approaches. Our key
contributions are as follows:
1. We demonstrate how European data can be adapted to provide the large volumes of data
required for ML models, given the fact that European Central Bank (ECB) zero-rate data
is only available from the early 2000s.
2. We benchmark traditional econometric frameworks against modern NN architectures,
exploring two distinct modeling strategies: (i) forecasting latent factors and mapping them
into zero rates, and (ii) forecasting individual rates directly.
3. The analysis is carried out for both the U.S. Treasury market and for bonds issued by the
ECB and compares both results. Furthermore, macroeconomic variables are incorporated
to enhance predictive performance.
3
4. A robust evaluation framework is proposed, which combines traditional statistical accuracy
metrics with an assessment of economic relevance through the performance of a bond
trading strategy based on each model’s forecasts.
This paper is structured as follows: Section 2 describes the data and the applied data
augmentation for the European data, outlines the model architectures, and details the resources
and methods employed. Section 3 focuses on the model performance and provides background
on the underlying investment strategy. Section 4 presents and discusses the empirical results.
Finally, Section 5 concludes the study by summarizing the key findings and their implications.
"""


result = chain.invoke({'text': text})

print(result)