import matplotlib.pyplot as plt

# lambdas = [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]
# errorTraining =[0.40735594114176166,0.25683045884298095,0.15927943848745443,0.10120922333582943,0.07834058111497944,0.06825269027434479,0.06566855922048778,0.06457767164929834,0.06381315929401245,0.06376870968121394,0.06375601082730438,0.06374966157314185]
# errorCV =[0.3496431687117006,0.2195135318819252,0.13568196401044322,0.08621360054541241,0.06691687833811844,0.05845209943917187,0.056286503386133134,0.05537360849582682,0.05473563439722421,0.05469821900949446,0.05468752989118822,0.0546821854976247]
lambdas = [100000000.0, 30000000.0, 10000000.0, 3000000.0, 1000000.0, 300000.0, 100000.0]
errorTraining =[0.2069382115629487,0.12895380111902394,0.08859662606472328,0.06609093272492499,0.05740027455049104,0.0540795138992588,0.05319087805165744]
errorCV =[0.20217220762174337,0.12133902833695084,0.0784383578088622,0.053292737409884425,0.04291717104532526,0.038608218975929164,0.037379721525211865]

plt.close('all')
plt.figure(1)
plt.plot(lambdas, errorTraining, '-o', label="training")
plt.plot(lambdas, errorCV, '-o', label="CV")
plt.xscale('log')
plt.ylabel("error")
plt.xlabel("lambda (regularization parameter)")

plt.legend(loc='upper left')
plt.show(block=False)

