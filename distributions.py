import collections as ct
import scipy.stats as st




income_model_dict = ct.OrderedDict()
income_model_dict['johnsonsu'] = st.johnsonsu(-5.3839367311065747,0.84376726932941271,-224.21280806585787,79.661998696081355)
income_model_dict['powerlaw'] = st.powerlaw(0.16342470577523971, -3.1423954341714262e-15, 55664716.096562646)
income_model_dict['exponpow'] = st.exponpow(0.25441022752240294, -1.8475789041433829e-22, 36120900.670255348)
income_model_dict['nakagami'] = st.nakagami(0.10038339454419823, -3.0390927147076284e-22, 33062195.426077582)
income_model_dict['exponweib'] = st.exponweib(-3.5157658448986489, 0.44492833350419714, -15427.454196748848, 2440.0278856175246)

drivingdistance_model_dict = ct.OrderedDict()
drivingdistance_model_dict['nakagami'] = st.nakagami(0.11928581143831021, 14.999999999999996, 41.404620910360876)
drivingdistance_model_dict['ncx2'] = st.ncx2(0.30254190304723211, 1.1286538320791935, 14.999999999999998, 8.7361471573932192)
drivingdistance_model_dict['chi'] = st.chi(0.47882729877571095, 14.999999999999996, 44.218301183844645)
drivingdistance_model_dict['recipinvgauss'] = st.recipinvgauss(2447246.0546641815, 14.999999999994969, 31.072009722580802)
drivingdistance_model_dict['f'] = st.f(0.85798489720127036, 4.1904554804436929, 14.99998319939356, 21.366492843433996)

drivingduration_model_dict = ct.OrderedDict()
drivingduration_model_dict['betaprime'] = st.betaprime(2.576282082814398, 9.7247974165209996, 9.1193851632305201, 261.3457987967214)
drivingduration_model_dict['exponweib'] = st.exponweib(2.6443841639764942, 0.89242254172118096, 10.603640861374947, 40.28556311444698)
drivingduration_model_dict['gengamma'] = st.gengamma(4.8743515108339581, 0.61806208678747043, 9.4649293818479716, 5.431576919220225)
drivingduration_model_dict['recipinvgauss'] = st.recipinvgauss(0.499908918842556, 0.78319699707613699, 28.725450197674746)
drivingduration_model_dict['f'] = st.f(9.8757694313677113, 12.347442183821462, 0.051160749890587665, 73.072591767722287)

carprice_model_dict = ct.OrderedDict()
carprice_model_dict['nct'] = st.nct(7.3139456577106312, 3.7415255108348946, -46.285705145385577, 7917.0860181436065)
carprice_model_dict['genlogistic'] = st.genlogistic(10.736440967148635, 3735.7049978006107, 10095.421377235754)
carprice_model_dict['gumbel_r'] = st.gumbel_r(26995.077239517472, 10774.370808211244)
carprice_model_dict['f'] = st.f(24168.523476867485, 35.805656864712923, -21087.314142557225, 51154.0328397044)
carprice_model_dict['johnsonsu'] = st.johnsonsu(-1.7479864366935538, 1.8675670208081987, 14796.793096897647, 14716.575397771712)

