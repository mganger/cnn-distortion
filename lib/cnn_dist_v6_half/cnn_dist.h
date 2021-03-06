
#pragma once
extern "C" {
#include <cblas.h>
};

#include <cmath>

struct cnn_dist_v6 {
	const static int latency = 511;
	const static int MAX_L = MAX_BUFFER + latency;
	const static int MAX_CH = 4;
	// About 8e-06*(MAX_BUFFER+511) MB of buffer
	float x_even[MAX_L][MAX_CH];
	float x_odd [MAX_L][MAX_CH];

	void operator()(float* x, float* y, int L) {

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {
			x_odd[i][0] = x[i];
		}

		// auto-generated code for layer layer_0: Conv1d(1, 4, kernel_size=(2,), stride=(1,), dilation=(512,))
		const float w_layer_0[2][1][4] = {{{0.4164464771747589,0.7477846145629883,0.5723923444747925,1.501248836517334}},{{1.4639004468917847,1.123586654663086,-0.8857517242431641,-0.7599462866783142}}};
		const float b_layer_0[4] = {0.003347989171743393,0.003884746227413416,0.008004417642951012,0.1215905249118805};

		// Fill with biases for layer_0
		for (int l = 256; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_0[i];
			}
		}

		// Apply main filter for layer_0
		// x_even[:,256:] = sum(w[k]@x_odd[:,256-(1-k)*256:L-(1-k)*256] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*256;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-256, 1, 1.0, &w_layer_0[k][0][0], 4, &x_odd[256-offset][0], MAX_CH, 1.0, &x_even[256][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 256; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_1: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(256,))
		const float w_layer_1[2][4][4] = {{{-0.18802504241466522,-0.854773759841919,-0.7215519547462463,-0.8421214818954468},{-0.2586285471916199,-0.3227439522743225,-0.35402771830558777,-0.2369672805070877},{-0.2897607982158661,0.41365760564804077,0.6298378705978394,-0.09161160886287689},{-0.3283819854259491,0.27256885170936584,0.22109659016132355,-0.22539950907230377}},{{0.415728360414505,0.022966932505369186,0.14345471560955048,0.1607455313205719},{0.4708334803581238,-0.2191057801246643,0.11248965561389923,-0.21490325033664703},{0.4417080581188202,0.018920307978987694,0.00620650127530098,-0.1686919629573822},{0.962893545627594,-0.3207530379295349,0.4818129241466522,0.007339727599173784}}};
		const float b_layer_1[4] = {-0.05623558536171913,0.2725513279438019,-0.08230441808700562,0.2399035394191742};

		// Fill with biases for layer_1
		for (int l = 384; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_1[i];
			}
		}

		// Apply main filter for layer_1
		// x_odd[:,384:] = sum(w[k]@x_even[:,384-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-384, 4, 1.0, &w_layer_1[k][0][0], 4, &x_even[384-offset][0], MAX_CH, 1.0, &x_odd[384][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 384; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_2: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(128,))
		const float w_layer_2[2][4][4] = {{{-0.03317662701010704,-0.002895868383347988,-0.41950586438179016,-0.1061168685555458},{-0.07163084298372269,-0.33264410495758057,-0.7276883125305176,-0.42040687799453735},{-0.7814343571662903,-0.5863019227981567,-0.11785975098609924,0.0724865272641182},{0.15188418328762054,-0.11427223682403564,0.2163659781217575,-0.22443197667598724}},{{0.8074720501899719,0.17644131183624268,0.01547366101294756,-0.4650214910507202},{-1.1089283227920532,0.976524293422699,0.5395373106002808,-0.19392795860767365},{-0.400789350271225,0.6458590626716614,0.15775616466999054,-1.1357029676437378},{-0.14467772841453552,0.48188939690589905,-0.10323715209960938,-0.24202494323253632}}};
		const float b_layer_2[4] = {0.2587038576602936,-0.1535634696483612,0.04826483875513077,0.25319889187812805};

		// Fill with biases for layer_2
		for (int l = 448; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_2[i];
			}
		}

		// Apply main filter for layer_2
		// x_even[:,448:] = sum(w[k]@x_odd[:,448-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-448, 4, 1.0, &w_layer_2[k][0][0], 4, &x_odd[448-offset][0], MAX_CH, 1.0, &x_even[448][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 448; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_3: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(64,))
		const float w_layer_3[2][4][4] = {{{0.2597988545894623,-0.6524063348770142,0.6650660634040833,0.10114424675703049},{-0.43268242478370667,-0.6358054876327515,-0.49653592705726624,-0.6482375264167786},{0.06345022469758987,-0.24123384058475494,0.16178114712238312,-0.1399543136358261},{0.28451305627822876,-0.3926936089992523,0.6595779061317444,0.17342323064804077}},{{-0.4959994852542877,-0.8840075135231018,-0.37019574642181396,0.22575025260448456},{0.8335776329040527,0.17878170311450958,0.7620079517364502,-0.05147184804081917},{0.4195229113101959,0.4951561689376831,0.7760641574859619,-0.2673892080783844},{-0.6471208333969116,0.24931639432907104,-0.3818316161632538,0.7314338684082031}}};
		const float b_layer_3[4] = {-0.2144152969121933,-0.027202637866139412,0.24224776029586792,0.07922152429819107};

		// Fill with biases for layer_3
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_3[i];
			}
		}

		// Apply main filter for layer_3
		// x_odd[:,480:] = sum(w[k]@x_even[:,480-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-480, 4, 1.0, &w_layer_3[k][0][0], 4, &x_even[480-offset][0], MAX_CH, 1.0, &x_odd[480][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_4: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(32,))
		const float w_layer_4[2][4][4] = {{{0.5193156599998474,-0.7765212059020996,0.22597935795783997,-0.253305047750473},{0.948691189289093,-0.22534047067165375,-0.16547849774360657,0.2861148715019226},{0.7068972587585449,-0.2927421033382416,-0.5282456278800964,-0.06180587410926819},{-0.04370284453034401,0.5920458436012268,0.9348170757293701,-0.4252614974975586}},{{-0.23404410481452942,-0.27010834217071533,0.8328246474266052,-0.37379467487335205},{0.5178858637809753,0.29698699712753296,0.3159700632095337,0.30548787117004395},{-0.23522484302520752,-0.44197842478752136,0.6005608439445496,-0.39485278725624084},{1.02493155002594,-0.25477299094200134,-0.43951836228370667,-0.6355786919593811}}};
		const float b_layer_4[4] = {-0.017650337889790535,-0.12525495886802673,0.23825488984584808,0.040755562484264374};

		// Fill with biases for layer_4
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_4[i];
			}
		}

		// Apply main filter for layer_4
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 4, 1.0, &w_layer_4[k][0][0], 4, &x_odd[496-offset][0], MAX_CH, 1.0, &x_even[496][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_5: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(16,))
		const float w_layer_5[2][4][4] = {{{-0.055269382894039154,0.07249671220779419,-0.42120426893234253,0.2772397994995117},{-0.3523944616317749,-0.5630910396575928,-0.3685670495033264,-0.08764734864234924},{-0.8149637579917908,-0.3782864212989807,-0.44615206122398376,-1.3697723150253296},{-0.07816636562347412,1.512674331665039,-0.19290772080421448,0.041055869311094284}},{{-0.15326273441314697,-0.8062439560890198,-0.45840945839881897,-0.12959420680999756},{-0.18461374938488007,2.145716667175293,-0.5123477578163147,0.3120633661746979},{-0.20604880154132843,0.4379376471042633,-0.11970875412225723,0.12585748732089996},{0.3065069019794464,0.42156678438186646,0.28469765186309814,0.6133463978767395}}};
		const float b_layer_5[4] = {-0.12773866951465607,-0.35182124376296997,0.08830584585666656,-0.08223827183246613};

		// Fill with biases for layer_5
		for (int l = 504; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_5[i];
			}
		}

		// Apply main filter for layer_5
		// x_odd[:,504:] = sum(w[k]@x_even[:,504-(1-k)*8:L-(1-k)*8] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*8;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-504, 4, 1.0, &w_layer_5[k][0][0], 4, &x_even[504-offset][0], MAX_CH, 1.0, &x_odd[504][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 504; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_6: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(8,))
		const float w_layer_6[2][4][4] = {{{0.0863654762506485,0.39443325996398926,0.0977238267660141,0.22236759960651398},{-0.4374677538871765,-0.664749264717102,-0.41556376218795776,0.030612552538514137},{0.6126754283905029,0.6528218388557434,0.6163425445556641,0.0016111296135932207},{-0.07558099925518036,0.5638340711593628,0.45912519097328186,0.5540890693664551}},{{0.2346513271331787,-0.47696471214294434,-0.33493855595588684,-0.4492335617542267},{0.3177480697631836,0.7194477319717407,-0.23087748885154724,0.4275164306163788},{0.03860535845160484,-0.2293556183576584,-0.3189201056957245,-0.38899558782577515},{-0.22817276418209076,0.1372283399105072,0.12570802867412567,-0.017333373427391052}}};
		const float b_layer_6[4] = {-0.055455856025218964,0.2457105666399002,-0.2864830493927002,0.33019381761550903};

		// Fill with biases for layer_6
		for (int l = 508; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_6[i];
			}
		}

		// Apply main filter for layer_6
		// x_even[:,508:] = sum(w[k]@x_odd[:,508-(1-k)*4:L-(1-k)*4] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*4;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-508, 4, 1.0, &w_layer_6[k][0][0], 4, &x_odd[508-offset][0], MAX_CH, 1.0, &x_even[508][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 508; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_7: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(4,))
		const float w_layer_7[2][4][4] = {{{0.35540279746055603,-0.030984338372945786,-0.13482873141765594,0.12152289599180222},{0.18400996923446655,0.2775675356388092,0.03521738201379776,0.8201605677604675},{0.019306257367134094,0.37297236919403076,-0.08130114525556564,0.6432942152023315},{0.022679463028907776,-0.377797394990921,0.6596900820732117,0.4722393751144409}},{{0.3785458207130432,-1.1337981224060059,-0.16619403660297394,0.01665145345032215},{0.05900299921631813,0.09628403931856155,-0.028950249776244164,0.7963458895683289},{-0.05576389282941818,0.6009237766265869,-0.12055414915084839,0.27389803528785706},{0.24535438418388367,-0.44019317626953125,0.2310594916343689,0.17105495929718018}}};
		const float b_layer_7[4] = {-0.2881080210208893,0.13069476187229156,-0.22298690676689148,0.01746486872434616};

		// Fill with biases for layer_7
		for (int l = 510; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_7[i];
			}
		}

		// Apply main filter for layer_7
		// x_odd[:,510:] = sum(w[k]@x_even[:,510-(1-k)*2:L-(1-k)*2] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*2;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-510, 4, 1.0, &w_layer_7[k][0][0], 4, &x_even[510-offset][0], MAX_CH, 1.0, &x_odd[510][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 510; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_8: Conv1d(4, 4, kernel_size=(2,), stride=(1,), dilation=(2,))
		const float w_layer_8[2][4][4] = {{{0.2925102114677429,-0.2760688066482544,-0.38255631923675537,-0.2655608355998993},{0.036261457949876785,0.21355989575386047,-0.29804179072380066,0.600396990776062},{-0.13748648762702942,-0.09498782455921173,0.3418809175491333,-0.4833495616912842},{0.5059455037117004,0.2423490732908249,0.12609703838825226,0.28627416491508484}},{{0.1827317476272583,-0.12781286239624023,-0.01409630011767149,0.29468902945518494},{-0.24066230654716492,-0.2386423647403717,0.4063117504119873,-0.43282967805862427},{0.383134663105011,-0.15360267460346222,0.2578367590904236,0.09074565768241882},{0.16737434267997742,0.028062263503670692,0.1871705800294876,-0.08907873183488846}}};
		const float b_layer_8[4] = {-0.3069136440753937,-0.16616511344909668,0.4772498905658722,-0.035868141800165176};

		// Fill with biases for layer_8
		for (int l = 511; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_8[i];
			}
		}

		// Apply main filter for layer_8
		// x_even[:,511:] = sum(w[k]@x_odd[:,511-(1-k)*1:L-(1-k)*1] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-511, 4, 1.0, &w_layer_8[k][0][0], 4, &x_odd[511-offset][0], MAX_CH, 1.0, &x_even[511][0], MAX_CH);
		}


		// Tanh (i.e. soft clip
		for (int l = 511; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = std::tanhf(v);
			}
		}


		// auto-generated code for layer layer_9: Conv1d(4, 1, kernel_size=(1,), stride=(1,))
		const float w_layer_9[1][4][1] = {{{0.451759397983551},{-0.0004368775407783687},{-0.3962245583534241},{0.2412981241941452}}};
		const float b_layer_9[1] = {0.2796299457550049};

		// Fill with biases for layer_9
		for (int l = 511; l < L; l++) {
			for (int i = 0; i < 1; i++) {
				x_odd[l][i] = b_layer_9[i];
			}
		}

		// Apply main filter for layer_9
		// x_odd[:,511:] = sum(w[k]@x_even[:,511-(0-k)*0:L-(0-k)*0] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, L-511, 4, 1.0, &w_layer_9[k][0][0], 1, &x_even[511-offset][0], MAX_CH, 1.0, &x_odd[511][0], MAX_CH);
		}


		// Copy result back to y
		for (int l = 511; l < L; l++) {
			y[l] = x_odd[l][0];
		}
	}
};
