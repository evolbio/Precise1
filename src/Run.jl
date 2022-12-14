module PPredict
using PPredict, Plots

S = Params(F=5.5, res_size=300, shift=0.5);
S = Params(F=5.5, res_size=30, shift=0.5, T=1000.0);

S = Params(F=5.5, res_size=50, shift=2.0, T=5000.0);
input_data, esn, mach, target_train, target_test, y_train, y_test = predict_driver(S);
best(mach)

# quick plot of Lorenz96
S=Params(N=125, F=4.0, T=10.0, fixed_init=false);
trj = PPredict.lorenz96(S);
pl=plot(size=(1500,900),legend=:none)
for i in 1:3 plot!(trj[:,i]) end
display(pl)

end # module PPredict
