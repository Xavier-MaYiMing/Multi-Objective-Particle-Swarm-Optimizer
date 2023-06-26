### Multi-Objective Particle Swarm Optimizer

##### Reference: Coello C,  Pulido G T,  Lechuga M S. Handling multiple objectives with particle swarm optimization[J]. IEEE Transactions on Evolutionary Computation.

The particle swarm optimizer for the global multi-objective optimization

| Variables  | Meaning                                                      |
| ---------- | ------------------------------------------------------------ |
| npop       | Population size                                              |
| iter       | Iteration number                                             |
| lb         | Lower bound                                                  |
| ub         | Upper bound                                                  |
| nrep       | Repository size                                              |
| ngrid      | Grid number on each dimension                                |
| alpha      | The inflation rate of grids                                  |
| beta       | The pressure of global best selection                        |
| gamma      | The pressure of deletion selection                           |
| mu         | Mutation rate                                                |
| omega      | Inertia weight (default = 0.5)                               |
| c1         | Personal learning coefficient (default = 1)                  |
| c2         | Global learning coefficient (default = 2)                    |
| dim        | Dimension                                                    |
| pos        | The positions of particles                                   |
| vel        | The velocities of particles                                  |
| objs       | The objectives of particles                                  |
| nobj       | Objective number                                             |
| vmin       | Minimum velocity                                             |
| vmax       | Maximum velocity                                             |
| pbest      | Personal best                                                |
| pbest_pos  | Personal best position                                       |
| rep        | The repository to store the positions of all non-dominated particles |
| rep_obj    | The repository to store the objectives of all non-dominated particles |
| grid       | The grid created on the objective space                      |
| grid_index | The grid index of all particles in the repository            |

#### Test problem: ZDT3



$$
\left\{
\begin{aligned}
&f_1(x)=x_1\\
&f_2(x)=g(x)\left[1-\sqrt{x_1/g(x)}-\frac{x_1}{g(x)}\sin(10\pi x_1)\right]\\
&f_3(x)=1+9\left(\sum_{i=2}^nx_i\right)/(n-1)\\
&x_i\in[0, 1], \qquad i=1,\cdots,n
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    main(200, 200, np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1]), 100, 7, 0.1, 2, 2, 0.1)
```

##### Output:

![MOPSO](C:\Users\dell\Desktop\研究生\个人算法主页\Multi-Objective Particle Swarm Optimization\MOPSO.gif)

```python
Iteration: 10, Pareto-optimal particles: 23
Iteration: 20, Pareto-optimal particles: 100
Iteration: 30, Pareto-optimal particles: 100
Iteration: 40, Pareto-optimal particles: 100
Iteration: 50, Pareto-optimal particles: 100
Iteration: 60, Pareto-optimal particles: 100
Iteration: 70, Pareto-optimal particles: 100
Iteration: 80, Pareto-optimal particles: 100
Iteration: 90, Pareto-optimal particles: 100
Iteration: 100, Pareto-optimal particles: 100
{
    'Pareto-optimal solutions': [
        [0.18302132403088703, 0.002084072873657628, 0, 0, 0], 
        [0.6307957796107608, 0, 0.002458174746413941, 0.0009019109847536464, 0.0021740342687729883], 
        [0.6425300876534454, 0.000606522927635565, 0.0008638835426194554, 0.0003597734852207002, 0.0013667024380544061], 
        [0.18500679183620838, 0.0005725476235495586, 0, 0.0006418432169176602, 0], 
        [0.6388356021799378, 0.0011782247529409482, 0, 0.000512920044259033, 0.003124923566250682], 
        [0.6439140864988065, 0.0006368121842343814, 2.005363077692469e-05, 0.0003819107463063023, 0], 
        [0.6230714496560541, 0.00045064767493511397, 0.004662467141578484, 0.0009778779471099341, 0.0004653065711093991], 
        [0.6282631865077398, 0.0006187082719303025, 0.0022977711963342706, 0.0001374365481989373, 0.004445158180497256], 
        [0.18366851691828018, 0.0013993647550026495, 0.000568608976863693, 0, 0.0003112839688256086], 
        [0.6259312795358801, 0.0012264226643225202, 0.0015387681604734549, 0.0006373521249254876, 0.0005879125503535452], 
        [0.6266181525112624, 0.0012041770129334, 0.0025216627864379434, 0.00027933887131226443, 0.0021602200363708673], 
        [0.18495835622088277, 0.0006039529842598547, 0, 0.00018917480031447026, 0.0008075398068034772], 
        [0.18260030216223772, 0.001248487984775658, 2.5381401622219503e-05, 0.00026199271613149423, 0.00024337573191352468], 
        [0.6220931323458017, 0.000808137571960506, 0.0011589024255222445, 0, 0.0018653275798473535], 
        [0.0025374014103080322, 0.0015389443654514653, 0.000444481015908703, 0.004098654929186622, 0.001113684782873306], 
        [0.634956129943436, 0.001108165287489852, 0.0013982310742466117, 0.0010398281605882972, 0.0015998460656266424], 
        [0.004990972900409271, 0.0011210517119440559, 0.00047170370528021766, 3.461028130932771e-05, 3.438293288211288e-05], 
        [0.07236534231595487, 0.0009120373531324756, 0, 0.00010264912321835207, 0], 
        [0.6403612525800797, 0.0005937964273581522, 5.628431147982607e-05, 0.0005753027812521612, 0.004763999282439986], 
        [0.06636771091285681, 0.0002753453126531293, 0, 0, 0.00017856088493138915], 
        [0.009594672985449693, 0.00036725598159302727, 0, 0.0020245425909376793, 0.0008497593930543154], 
        [0.0006779081458873869, 0.0011375857802681086, 0.0006371304792445676, 0.0003292355923428975, 0.003147671288341153], 
        [0.007929288851526092, 6.130535682718844e-05, 0, 0.0010928868342906122, 0.0007028625922452251], 
        [0.022979073656974935, 0.00011730120045287098, 0, 0.00048636283737431247, 6.831843391780419e-05], 
        [0.44698747854505017, 0.001210798141100956, 0.00048710705615666555, 0.00033135979169153875, 0.0015283564802173139], 
        [0.6220652953999579, 0.0011238462825333985, 0.0025981658880086227, 0.0006411062616766394, 0.0017408835552804637], 
        [0.01965680003075357, 0.000988559346847454, 0, 0.0008137040657284061, 0.0005365496002746779], 
        [0.0066899145553345335, 9.64084320171673e-05, 0, 0.0011200016422401302, 0.0006325139015711566], 
        [0.050499454578052265, 0.0007402791947910082, 0.0008613496116294081, 0.00028072247733139247, 0], 
        [0.413293184165735, 0.000561125663172776, 0, 0.0006576492962615659, 0.0008525198797196968], 
        [0.6356462232192017, 0.000516924625767301, 0.0013357120468328453, 0.0003377484403031108, 0.004833125422568207], 
        [0.20215230106534238, 0.0013080269320505582, 0.0007976748608294968, 0.000303278197465047, 0], 
        [0.44455096215087264, 0.00039672958949255, 0.0018796879572419896, 0.0010110064932124267, 0.0005560671449149073], 
        [0.24552408688272245, 0.0011079035718227635, 0.0005931618957359139, 0.00043320552215199014, 0], 
        [0.011883014338158082, 0.0006617461570408049, 0.0010489908188482384, 0.0007818946961230876, 0.0013313304682298433], 
        [0.4451285083131187, 0.0008506216319346001, 0.0008545260652317638, 0.0007739479726734019, 0], 
        [0, 0, 0, 0.0002704283435119003, 0], 
        [0.18325024505575155, 0.0004047540398602917, 0.0007899905185879078, 0.0002864126569715745, 0], 
        [0.41289547951548133, 0.0007919380258844453, 0.00048049432805102996, 0.00020885771676439958, 0.00037244393504452133], 
        [0.05555186042797341, 0.0006394780539908215, 0.0009123861905632456, 0.0003295656543255077, 0.00011230935427894152], 
        [0.637060098867164, 0.0010108629932756715, 0.001331572411671745, 0.0003414930416608072, 0.002879623581169069], 
        [0.41545340184861146, 0.0012976940492455168, 0.0005653127701677892, 0.000702951762604023, 0.001409873378199696], 
        [0.6509239723854794, 0.0009622916653059887, 0.0008396028736302705, 0.00035602306569541337, 0.002287121248129017], 
        [0.4481153848558467, 0.0012033678761778784, 0.0005736928552352674, 0.0014808013945387388, 0.0020579290215105668], 
        [0.009936254687876692, 0.0012960708016321173, 3.7314490813397244e-05, 0.0012207068687371775, 0.0009132658876006828], 
        [0.0027520850458427337, 0.00021497789405680899, 0.00038853406815307504, 0.0016100855455906955, 0], 
        [0.01576708788730043, 0.00020763359656865855, 9.634043327169678e-05, 0.0014718156042151124, 0.0006572920750924067], 
        [0.4160746820439195, 0.0010053352530591641, 1.7742301022533188e-06, 0.0008369786208402893, 0.0014519813408144866], 
        [0.6340031469875357, 0, 0.002145350230219817, 0.000556369908961204, 0.0017551121466529863], 
        [0.41579602783047087, 0.0008705003110058324, 0.0011319407674799345, 0.0008263741635102864, 0.0006337344353620424], 
        [0.2208854529201403, 0.0003505718757033439, 0.0006277825633023429, 0.0001366925420405069, 0.0007615296743983726], 
        [0.2561963499350465, 0.0008514734128022749, 4.2264781100102974e-05, 0.0007685346686208774, 0], 
        [0.20865125573011145, 0.0004738237294028995, 0.0020299478505784226, 0.0008290934816082972, 0.00046528932482518197], 
        [0.44630148044784146, 0.0007463558370209536, 0, 0.0002851556488411471, 0.0012439460651377584], 
        [0.41509515862596835, 0.00017310987804735754, 0.0002995935830648073, 0.000624620498525742, 0.0003334005650328048], 
        [0.6410452388331315, 0, 0.00316590069587563, 0.0010004162807142757, 0.0015290377216343946], 
        [0.41261362568493887, 0.0006590916014374403, 0.0014447400626041434, 0.0002647483041244691, 0.0026355578779711075], 
        [0.027538004874341687, 0, 0, 0.0011905329650835383, 0.00018491038612862115], 
        [0.43532723714044774, 0.0006994094724920001, 0.0021240720133064005, 0.0005849872677074976, 0.0006901041720046617], 
        [0.07107281586281397, 3.996939207686755e-05, 0, 0.0002762241730719761, 0.0008163459834986287], 
        [0.42236404825424145, 0.0006738468454455425, 0, 0.0009030647551799761, 0.0007175465941660412], 
        [0.1864474113935112, 0.0007132444799282393, 0.000913755063024563, 0.00014450106477379376, 0.0016723581000065875], 
        [0.19331143413539048, 0.0006230806097013639, 0, 0.0002369839923710211, 0], 
        [0.434837522069788, 0.00114539310954298, 0.0008031694081555927, 0.000309864004666637, 0.0012537654121999794], 
        [0.19764834827875785, 0.0007053907851694946, 0.0004648815235874772, 0.00044405906178317375, 0.005362890056522134], 
        [0.07088545292014031, 0.00137346278451328, 0.001063111953342523, 0.00037509621033845087, 0.0011586982448913842], 
        [0.23785522159829525, 0.0007703399351014111, 0, 0.0014329073063110032, 0.000980915452108777], 
        [0.6273229443851702, 0.0007469682172714958, 0.004254998049854011, 0.0005046437594059773, 0.00036399847543970733], 
        [0.4490745008384732, 0.0007341620344869456, 0.00135325562348343, 0.0007998204497482014, 0.0010193178705470692], 
        [0.2423718142075547, 0.0005561967117660129, 0, 0.0016224724736615234, 0.00013292735557549464], 
        [0.4343216393772592, 0.0010502316592436904, 0, 0.0006845307058611148, 0.0009571099550508068], 
        [0.4435677059302108, 0.0009300261563753067, 0.0005277786766821303, 0.0005199567795849114, 0], 
        [0.18863544894380047, 0.0005573144490305904, 0.0003617176210493787, 0.00019257900245667488, 0.00129071412968646], 
        [0.04357854375788363, 0.00010381206568687012, 0, 0.0002871287954122368, 0.0008523219825734661], 
        [0.0807924102590669, 0.0009849668922998563, 0.0005127724092780996, 0.0009278056500911571, 0], 
        [0.41432609766491346, 0.0010431626178706676, 0.0012218460828398749, 0.0005513333867971157, 0.0011136911427041497], 
        [0.4472480230475657, 0.0008563865368309674, 0.0007727679046647323, 8.962045183032324e-05, 0.0021688705226301295], 
        [0.43051673910697164, 0.00013718367220682647, 0.0015499593905984568, 0.0002792176518743439, 0.0008505090806540591], 
        [0.030212077774338153, 0.0015618854504130043, 0.00099458536105623, 0.0012204012851845009, 0.0009791957780480218], 
        [0.22283787513589184, 0.0010652148642407995, 0.00023112337016667798, 0.0003101901231240807, 0.001132353216555181], 
        [0.18719977831190904, 0.00022122334284780418, 0.0007104825708083133, 0.0006662106973569433, 0.0026337539714737716], 
        [0.23957929338115772, 0.0007045496783417228, 0.00033256751896305566, 9.781930469357492e-05, 0.0009172438338109909], 
        [0.4483561120261571, 0.0010737752946010587, 0.0007365332869098348, 0.0007138701957667859, 0.0013389313140597635], 
        [0.42525556962779976, 0.0010489232264873213, 0.0010909041864945101, 0.0013163795015224452, 0], 
        [0.2263120773754268, 0.0004683516010633435, 0.001283024503111573, 0.0014011752176634896, 0.0001247979917958134], 
        [0.0478083179830108, 2.792521717026704e-05, 0, 0.0010150658301259973, 0.0007664762356258102], 
        [0.24035296803771197, 0.00014923750543248415, 0.0004083518956963676, 7.03822530999051e-05, 0.00018169070913359508], 
        [0.08047526770774133, 0.0014851390513093672, 2.7375663891124436e-05, 0.001036518384728529, 0.0011735063860635114], 
        [0.44186265885572995, 0.0006822680750616226, 0, 0.0005308848950248335, 0.0015931801621797905], 
        [0.18599584089346383, 0.0013297921233926158, 0.00041827450018097525, 0.0006131946778982984, 0.0012599041886695254], 
        [0.20149857634991095, 0.0005899534398462646, 0.00041775674202290615, 0.000651533345422671, 0], 
        [0.1891807601481871, 0.00040788019284921593, 0.0009070706984948068, 0.0010159753947815463, 0.0017009811179630357], 
        [0.205017709992358, 0.0005893013992837317, 0, 0.002766949185937271, 0.0006810150222350176], 
        [0.01586485360406123, 0.0005309777658380632, 9.83360716315493e-05, 4.352914544389813e-05, 0.00010226370198299409], 
        [0.1894170299441768, 0, 0.0008156225958329717, 0.00169910846913494, 0.0013847368449590091], 
        [0.042842507840170546, 0.00024995817724272006, 0, 0.001162986045337356, 0.00012575797271326562], 
        [0.4126452781873937, 0.0007420666671308308, 0.0013953528949744816, 0.0016596361832961308, 0], 
        [0.4312930770154909, 0.0006615710603280693, 0.001858272962871375, 0.0011675126791651413, 0.0008210177024486707], 
        [0.06021690970441176, 0.00029783815859276655, 0.0003463771555505369, 0.000539654993660234, 0.0009098703384169745], 
        [0.44550672816893244, 0.0010798550325328423, 0.0012072925689001082, 0.0008571950165867616, 0.00023663446373353502]], 
    'Pareto points': [
        [0.18302132403088703, 0.6689372645163523], 
        [0.6307957796107608, -0.3061376630319624], 
        [0.6425300876534454, -0.4221829014776697], 
        [0.18500679183620838, 0.6559771417045296], 
        [0.6388356021799378, -0.3926971035814375], 
        [0.6439140864988065, -0.43322316743968575], 
        [0.6230714496560541, -0.1934904651795885], 
        [0.6282631865077398, -0.269796360571025], 
        [0.18366851691828018, 0.6656194344657907], 
        [0.6259312795358801, -0.2410815953394972], 
        [0.6266181525112624, -0.24822009774142365], 
        [0.18495835622088277, 0.6569447699457489], 
        [0.18260030216223772, 0.6707485595356012], 
        [0.6220931323458017, -0.1814370501592704], 
        [0.0025374014103080322, 0.9652096505420298], 
        [0.634956129943436, -0.35521481321755644], 
        [0.004990972900409271, 0.9321808055269408], 
        [0.07236534231595487, 0.6777427066554523], 
        [0.6403612525800797, -0.4033498192386333], 
        [0.06636771091285681, 0.6854852107410112], 
        [0.009594672985449693, 0.9061360609909037], 
        [0.0006779081458873869, 0.9856116715389593], 
        [0.007929288851526092, 0.9129911099876837], 
        [0.022979073656974935, 0.8346239059392633], 
        [0.44698747854505017, -0.10822422925549155], 
        [0.6220652953999579, -0.1778764909042713], 
        [0.01965680003075357, 0.8533093593697136], 
        [0.0066899145553345335, 0.920802536972743], 
        [0.050499454578052265, 0.7285458390340507], 
        [0.413293184165735, 0.19266016789560625], 
        [0.6356462232192017, -0.35985270432438643], 
        [0.20215230106534238, 0.5409314268029735], 
        [0.44455096215087264, -0.09902855837908953], 
        [0.24552408688272245, 0.26500916778758954], 
        [0.011883014338158082, 0.8947929322877318], 
        [0.4451285083131187, -0.10338549524942589], 
        [0, 1.0006084637729018], 
        [0.18325024505575155, 0.666581570916814], 
        [0.41289547951548133, 0.19752740612592287], 
        [0.05555186042797341, 0.7135544084820767], 
        [0.637060098867164, -0.3757641040713191], 
        [0.41545340184861146, 0.16764698626836036], 
        [0.6509239723854794, -0.4514715119287184], 
        [0.4481153848558467, -0.10877524412061147], 
        [0.009936254687876692, 0.9046811738535381], 
        [0.0027520850458427337, 0.9521521490829242], 
        [0.01576708788730043, 0.8720695829149855], 
        [0.4160746820439195, 0.15868980718221587], 
        [0.6340031469875357, -0.3458090136892623], 
        [0.41579602783047087, 0.1624889843640204], 
        [0.2208854529201403, 0.3984933119569689], 
        [0.2561963499350465, 0.24527853451170228], 
        [0.20865125573011145, 0.49380144510222584], 
        [0.44630148044784146, -0.10793836587181037], 
        [0.41509515862596835, 0.16835007777217478], 
        [0.6410452388331315, -0.408797385379393], 
        [0.41261362568493887, 0.20604298114310035], 
        [0.027538004874341687, 0.815930975168556], 
        [0.43532723714044774, -0.043497444151511924], 
        [0.07107281586281397, 0.6795551607240536], 
        [0.42236404825424145, 0.0806624406804383], 
        [0.1864474113935112, 0.6512898451584889], 
        [0.19331143413539048, 0.6021597608629738], 
        [0.434837522069788, -0.04055042908969283], 
        [0.19764834827875785, 0.5822349404324256], 
        [0.07088545292014031, 0.6853370555651576], 
        [0.23785522159829525, 0.2969647571495678], 
        [0.6273229443851702, -0.2587886972015343], 
        [0.4490745008384732, -0.11316385124535312], 
        [0.2423718142075547, 0.27616445371157666], 
        [0.4343216393772592, -0.037660622324244726], 
        [0.4435677059302108, -0.09758099751472406], 
        [0.18863544894380047, 0.6358375933087622], 
        [0.04357854375788363, 0.751056040567203], 
        [0.0807924102590669, 0.6745959684473745], 
        [0.41432609766491346, 0.18207845860608957], 
        [0.4472480230475657, -0.10851494903212697], 
        [0.43051673910697164, -0.0042330688350245395], 
        [0.030212077774338153, 0.8113975350118886], 
        [0.22283787513589184, 0.38614106645214186], 
        [0.18719977831190904, 0.648066743059515], 
        [0.23957929338115772, 0.2871652164046259], 
        [0.4483561120261571, -0.11156408412135135], 
        [0.42525556962779976, 0.05002387726192062], 
        [0.2263120773754268, 0.3634152394610925], 
        [0.0478083179830108, 0.7372804860442533], 
        [0.24035296803771197, 0.2817185320736787], 
        [0.08047526770774133, 0.6771836969919117], 
        [0.44186265885572995, -0.0880112031639959], 
        [0.18599584089346383, 0.6543370795909732], 
        [0.20149857634991095, 0.5445273574087758], 
        [0.1891807601481871, 0.6352250872287766], 
        [0.205017709992358, 0.5220586572772817], 
        [0.01586485360406123, 0.8680945882156568], 
        [0.1894170299441768, 0.6334704120382124], 
        [0.042842507840170546, 0.7543558043989751], 
        [0.4126452781873937, 0.20377963557491452], 
        [0.4312930770154909, -0.00883873476786799], 
        [0.06021690970441176, 0.701600694019245], 
        [0.44550672816893244, -0.10346511599444425]
    ]
}

```
