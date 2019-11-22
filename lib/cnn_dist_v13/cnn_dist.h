
#pragma once
extern "C" {
#include <cblas.h>
};

#include <cmath>

struct cnn_dist_v13 {
	const static int latency = 748;
	const static int MAX_L = MAX_BUFFER + latency;
	const static int MAX_CH = 4;
	// About 8e-06*(MAX_BUFFER+748) MB of buffer
	float x_even[MAX_L][MAX_CH];
	float x_odd [MAX_L][MAX_CH];

	void operator()(float* x, float* y, int L) {

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {
			x_odd[i][0] = x[i];
		}

		// auto-generated code for layer layer_0: Conv1d(1, 3, kernel_size=(2,), stride=(1,), dilation=(256,), bias=False)
		const float w_layer_0[2][1][3] = {{{0.12404585629701614,1.240994930267334,-0.6440662145614624}},{{0.9711227416992188,0.44973236322402954,0.9789575934410095}}};

		// Apply main filter for layer_0
		// x_even[:,256:] = sum(w[k]@x_odd[:,256-(1-k)*256:L-(1-k)*256] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*256;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-256, 1, 1.0, &w_layer_0[k][0][0], 3, &x_odd[256-offset][0], MAX_CH, beta, &x_even[256][0], MAX_CH);
		}


		// auto-generated code for layer layer_1: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_1[2][3][3] = {{{-0.5373702645301819,-0.26802223920822144,1.4378767013549805},{-0.18228600919246674,0.25675898790359497,0.4628697633743286},{0.08329009264707565,0.4027999937534332,1.1372591257095337}},{{0.10790754109621048,-0.23131877183914185,0.10278520733118057},{-1.4135996103286743,-1.2301385402679443,-0.3772827386856079},{0.7041118144989014,0.5883483290672302,0.12406588345766068}}};

		// Apply main filter for layer_1
		// x_odd[:,384:] = sum(w[k]@x_even[:,384-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-384, 3, 1.0, &w_layer_1[k][0][0], 3, &x_even[384-offset][0], MAX_CH, beta, &x_odd[384][0], MAX_CH);
		}


		// auto-generated code for layer layer_2: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_2[2][3][3] = {{{-0.06924855709075928,-0.015299374237656593,-0.7339816093444824},{-0.45117753744125366,-0.3129070997238159,-0.2815137803554535},{-1.3396753072738647,-1.0098943710327148,1.0368269681930542}},{{-0.22445297241210938,-1.0395796298980713,-0.5194834470748901},{-0.482441782951355,-0.9428719878196716,-0.4589847922325134},{0.15095144510269165,-0.07837019115686417,-0.06979356706142426}}};

		// Apply main filter for layer_2
		// x_even[:,448:] = sum(w[k]@x_odd[:,448-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-448, 3, 1.0, &w_layer_2[k][0][0], 3, &x_odd[448-offset][0], MAX_CH, beta, &x_even[448][0], MAX_CH);
		}


		// auto-generated code for layer layer_3: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_3[2][3][3] = {{{-0.7240022420883179,0.6566280126571655,-1.238822102546692},{-1.1533955335617065,0.07400931417942047,0.00801421795040369},{0.7040629386901855,-0.8221089243888855,1.1146090030670166}},{{-0.0732446238398552,-0.3196682929992676,0.41447606682777405},{-0.7704986333847046,-0.20640996098518372,-0.7861284613609314},{-0.8452701568603516,0.37358033657073975,0.3193286061286926}}};

		// Apply main filter for layer_3
		// x_odd[:,480:] = sum(w[k]@x_even[:,480-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-480, 3, 1.0, &w_layer_3[k][0][0], 3, &x_even[480-offset][0], MAX_CH, beta, &x_odd[480][0], MAX_CH);
		}


		// auto-generated code for layer layer_4: Conv1d(3, 2, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_4[2][3][2] = {{{-0.7143328189849854,-1.3611496686935425},{0.6274238228797913,0.42945295572280884},{-0.6252164840698242,-0.9794629812240601}},{{-0.9385376572608948,-0.592017650604248},{-0.569079577922821,-0.3765937089920044},{0.8704333901405334,-0.09206411242485046}}};

		// Apply main filter for layer_4
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-496, 3, 1.0, &w_layer_4[k][0][0], 2, &x_odd[496-offset][0], MAX_CH, beta, &x_even[496][0], MAX_CH);
		}


		// auto-generated code for layer layer_5: Conv1d(2, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_5[1][2][4] = {{{-0.5488660335540771,1.4547024965286255,-1.5829682350158691,-1.1512260437011719},{-0.4999045729637146,1.2225542068481445,-0.34881046414375305,1.6321176290512085}}};
		const float b_layer_5[4] = {-0.47964999079704285,-0.1687014400959015,0.017721161246299744,-0.1587008684873581};

		// Fill with biases for layer_5
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_5[i];
			}
		}

		// Apply main filter for layer_5
		// x_odd[:,496:] = sum(w[k]@x_even[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 2, 1.0, &w_layer_5[k][0][0], 4, &x_even[496-offset][0], MAX_CH, 1.0, &x_odd[496][0], MAX_CH);
		}


		// Simple Tanh (polynomial)
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = v > 2 ? 1.3333333333333333f : v < -2 ? -1.3333333333333333f : v - v*v*v*0.08333333333333333f;
			}
		}


		// auto-generated code for layer layer_6: Conv1d(4, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_6[1][4][4] = {{{-0.18213212490081787,-0.7949153184890747,0.5164827704429626,0.07614351809024811},{0.9127683639526367,0.22535385191440582,-0.017431652173399925,0.8261185884475708},{0.3372681438922882,-0.4599730670452118,0.581878125667572,-0.5667939186096191},{1.591963291168213,-0.02573980763554573,-0.08488685637712479,-0.09391283988952637}}};
		const float b_layer_6[4] = {-0.6037859916687012,0.44285598397254944,-0.25858074426651,0.1947997361421585};

		// Fill with biases for layer_6
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_6[i];
			}
		}

		// Apply main filter for layer_6
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 4, 1.0, &w_layer_6[k][0][0], 4, &x_odd[496-offset][0], MAX_CH, 1.0, &x_even[496][0], MAX_CH);
		}


		// Simple Tanh (polynomial)
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = v > 2 ? 1.3333333333333333f : v < -2 ? -1.3333333333333333f : v - v*v*v*0.08333333333333333f;
			}
		}


		// auto-generated code for layer layer_7: Conv1d(4, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_7[1][4][4] = {{{0.5042926073074341,-0.04929687827825546,-0.06666484475135803,0.5141434073448181},{-0.613980233669281,0.004275052342563868,-0.2729881703853607,0.49391230940818787},{0.20911617577075958,0.20719249546527863,-0.20788760483264923,-0.3453318178653717},{0.4262348711490631,0.2319493591785431,0.14678028225898743,0.44664037227630615}}};
		const float b_layer_7[4] = {0.5412876009941101,0.06203816086053848,0.025419466197490692,0.4713478088378906};

		// Fill with biases for layer_7
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_7[i];
			}
		}

		// Apply main filter for layer_7
		// x_odd[:,496:] = sum(w[k]@x_even[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 4, 1.0, &w_layer_7[k][0][0], 4, &x_even[496-offset][0], MAX_CH, 1.0, &x_odd[496][0], MAX_CH);
		}


		// Simple Tanh (polynomial)
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = v > 2 ? 1.3333333333333333f : v < -2 ? -1.3333333333333333f : v - v*v*v*0.08333333333333333f;
			}
		}


		// auto-generated code for layer layer_8: Conv1d(4, 2, kernel_size=(1,), stride=(1,))
		const float w_layer_8[1][4][2] = {{{-0.37979379296302795,-0.3750858008861542},{-0.0016147223068401217,-0.010641785338521004},{-0.003479798324406147,-0.0053815580904483795},{0.5694546699523926,-0.3281669020652771}}};
		const float b_layer_8[2] = {-0.2829188406467438,0.033284611999988556};

		// Fill with biases for layer_8
		for (int l = 496; l < L; l++) {
			for (int i = 0; i < 2; i++) {
				x_even[l][i] = b_layer_8[i];
			}
		}

		// Apply main filter for layer_8
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-496, 4, 1.0, &w_layer_8[k][0][0], 2, &x_odd[496-offset][0], MAX_CH, 1.0, &x_even[496][0], MAX_CH);
		}


		// auto-generated code for layer layer_9: Conv1d(2, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_9[2][2][3] = {{{-0.1009327694773674,-0.04430904984474182,0.5193083882331848},{0.13850116729736328,-0.14205026626586914,-0.39934009313583374}},{{0.668626070022583,0.17307277023792267,-0.08881682902574539},{-0.42278775572776794,0.316179484128952,-0.2800169885158539}}};

		// Apply main filter for layer_9
		// x_odd[:,624:] = sum(w[k]@x_even[:,624-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-624, 2, 1.0, &w_layer_9[k][0][0], 3, &x_even[624-offset][0], MAX_CH, beta, &x_odd[624][0], MAX_CH);
		}


		// auto-generated code for layer layer_10: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_10[2][3][3] = {{{-0.156342551112175,0.12369997799396515,0.4465814232826233},{-0.1898653656244278,0.15215031802654266,-0.023110179230570793},{0.026897817850112915,-0.08886687457561493,0.4649260640144348}},{{0.29468366503715515,0.8168556094169617,-0.4506395757198334},{0.7723485827445984,-0.09376326948404312,0.04555178061127663},{0.32473626732826233,0.010769439861178398,0.35710856318473816}}};

		// Apply main filter for layer_10
		// x_even[:,688:] = sum(w[k]@x_odd[:,688-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-688, 3, 1.0, &w_layer_10[k][0][0], 3, &x_odd[688-offset][0], MAX_CH, beta, &x_even[688][0], MAX_CH);
		}


		// auto-generated code for layer layer_11: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_11[2][3][3] = {{{-0.17557375133037567,-0.387597918510437,-0.3155759572982788},{-0.5752955079078674,-0.2574384808540344,-0.40786027908325195},{0.25759220123291016,-0.37227684259414673,0.310972660779953}},{{0.02459930069744587,-0.21066978573799133,0.3558814227581024},{0.2022819072008133,0.003354271175339818,-0.08982717245817184},{0.5529116988182068,-0.2102869302034378,0.15405836701393127}}};

		// Apply main filter for layer_11
		// x_odd[:,720:] = sum(w[k]@x_even[:,720-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-720, 3, 1.0, &w_layer_11[k][0][0], 3, &x_even[720-offset][0], MAX_CH, beta, &x_odd[720][0], MAX_CH);
		}


		// auto-generated code for layer layer_12: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_12[2][3][3] = {{{-0.07523353397846222,-0.11134215444326401,-0.3630828857421875},{0.009077307768166065,0.3889577090740204,-0.953948974609375},{-0.038812872022390366,-0.28794291615486145,-0.4654713273048401}},{{-0.2405954897403717,-0.5994563698768616,0.031199494376778603},{0.28904014825820923,-0.19792839884757996,0.4269162118434906},{0.4128532111644745,-0.18574446439743042,0.37406593561172485}}};

		// Apply main filter for layer_12
		// x_even[:,736:] = sum(w[k]@x_odd[:,736-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-736, 3, 1.0, &w_layer_12[k][0][0], 3, &x_odd[736-offset][0], MAX_CH, beta, &x_even[736][0], MAX_CH);
		}


		// auto-generated code for layer layer_13: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(8,), bias=False)
		const float w_layer_13[2][3][3] = {{{-0.8562594056129456,-0.44259390234947205,-0.1286805421113968},{0.19521456956863403,0.08809155970811844,-0.24249769747257233},{-0.07911063730716705,0.12326279282569885,0.5113371014595032}},{{0.36099308729171753,0.3685988485813141,0.12452743947505951},{-0.09258703887462616,-0.08992110937833786,-0.36161860823631287},{-0.3730052411556244,-0.17242564260959625,-0.5990736484527588}}};

		// Apply main filter for layer_13
		// x_odd[:,744:] = sum(w[k]@x_even[:,744-(1-k)*8:L-(1-k)*8] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*8;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-744, 3, 1.0, &w_layer_13[k][0][0], 3, &x_even[744-offset][0], MAX_CH, beta, &x_odd[744][0], MAX_CH);
		}


		// auto-generated code for layer layer_14: Conv1d(3, 1, kernel_size=(2,), stride=(1,), dilation=(4,))
		const float w_layer_14[2][3][1] = {{{-0.34674298763275146},{-0.10990015417337418},{0.1542004942893982}},{{0.34969615936279297},{0.10603734850883484},{0.31334182620048523}}};
		const float b_layer_14[1] = {-0.002300877124071121};

		// Fill with biases for layer_14
		for (int l = 748; l < L; l++) {
			for (int i = 0; i < 1; i++) {
				x_even[l][i] = b_layer_14[i];
			}
		}

		// Apply main filter for layer_14
		// x_even[:,748:] = sum(w[k]@x_odd[:,748-(1-k)*4:L-(1-k)*4] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*4;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, L-748, 3, 1.0, &w_layer_14[k][0][0], 1, &x_odd[748-offset][0], MAX_CH, 1.0, &x_even[748][0], MAX_CH);
		}


		// Copy result back to y
		for (int l = 748; l < L; l++) {
			y[l] = x_even[l][0];
		}
	}
};
