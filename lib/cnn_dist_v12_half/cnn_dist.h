
#pragma once
extern "C" {
#include <cblas.h>
};

#include <cmath>

struct cnn_dist_v12 {
	const static int latency = 374;
	const static int MAX_L = MAX_BUFFER + latency;
	const static int MAX_CH = 4;
	// About 8e-06*(MAX_BUFFER+374) MB of buffer
	float x_even[MAX_L][MAX_CH];
	float x_odd [MAX_L][MAX_CH];

	void operator()(float* x, float* y, int L) {

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {
			x_odd[i][0] = x[i];
		}

		// auto-generated code for layer layer_0: Conv1d(1, 3, kernel_size=(2,), stride=(1,), dilation=(256,), bias=False)
		const float w_layer_0[2][1][3] = {{{-0.08583030849695206,0.5186648368835449,-0.8003621101379395}},{{-1.0631130933761597,1.0840036869049072,0.8071127533912659}}};

		// Apply main filter for layer_0
		// x_even[:,128:] = sum(w[k]@x_odd[:,128-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-128, 1, 1.0, &w_layer_0[k][0][0], 3, &x_odd[128-offset][0], MAX_CH, beta, &x_even[128][0], MAX_CH);
		}


		// auto-generated code for layer layer_1: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_1[2][3][3] = {{{0.2636786699295044,1.0707275867462158,-0.28979936242103577},{0.010127183049917221,-1.2835906744003296,0.24459198117256165},{-0.7445932030677795,-0.7622990608215332,-0.005548056680709124}},{{-0.03021678701043129,-0.2614839971065521,0.4230376183986664},{-0.28591257333755493,-0.26638010144233704,-0.26542794704437256},{-0.031202159821987152,0.04599398747086525,0.7870035767555237}}};

		// Apply main filter for layer_1
		// x_odd[:,192:] = sum(w[k]@x_even[:,192-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-192, 3, 1.0, &w_layer_1[k][0][0], 3, &x_even[192-offset][0], MAX_CH, beta, &x_odd[192][0], MAX_CH);
		}


		// auto-generated code for layer layer_2: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_2[2][3][3] = {{{0.2764991819858551,0.619861364364624,0.42969435453414917},{-0.09734676778316498,0.6869813799858093,0.5311681628227234},{-0.18309541046619415,0.029998496174812317,0.18954531848430634}},{{-0.34401893615722656,0.018978215754032135,-0.10648131370544434},{-1.0063894987106323,0.29526472091674805,-0.63907790184021},{0.16109558939933777,0.9205557107925415,0.6154322028160095}}};

		// Apply main filter for layer_2
		// x_even[:,224:] = sum(w[k]@x_odd[:,224-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-224, 3, 1.0, &w_layer_2[k][0][0], 3, &x_odd[224-offset][0], MAX_CH, beta, &x_even[224][0], MAX_CH);
		}


		// auto-generated code for layer layer_3: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_3[2][3][3] = {{{0.39770349860191345,-0.7468500733375549,-0.5976365804672241},{-0.8676868081092834,0.19269534945487976,0.2044781595468521},{-0.03558063134551048,-0.985572099685669,-0.4192846119403839}},{{0.2553304135799408,-0.12198462337255478,0.4148417115211487},{-0.6577630639076233,-0.4945431351661682,-0.5318288803100586},{-0.5886655449867249,-0.10778156667947769,-0.44109970331192017}}};

		// Apply main filter for layer_3
		// x_odd[:,240:] = sum(w[k]@x_even[:,240-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-240, 3, 1.0, &w_layer_3[k][0][0], 3, &x_even[240-offset][0], MAX_CH, beta, &x_odd[240][0], MAX_CH);
		}


		// auto-generated code for layer layer_4: Conv1d(3, 2, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_4[2][3][2] = {{{-0.6477870345115662,-0.610987663269043},{-0.10562854260206223,0.5860682129859924},{-0.2741983234882355,0.2380448579788208}},{{0.4446178674697876,-0.04853849858045578},{-0.9076033234596252,-1.0434296131134033},{-0.7311594486236572,-0.7338244915008545}}};

		// Apply main filter for layer_4
		// x_even[:,248:] = sum(w[k]@x_odd[:,248-(1-k)*8:L-(1-k)*8] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*8;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-248, 3, 1.0, &w_layer_4[k][0][0], 2, &x_odd[248-offset][0], MAX_CH, beta, &x_even[248][0], MAX_CH);
		}


		// auto-generated code for layer layer_5: Conv1d(2, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_5[1][2][4] = {{{-0.5009680390357971,0.8821141719818115,-1.702233076095581,0.47020605206489563},{-1.5255439281463623,0.39772993326187134,-0.1289607584476471,0.9674017429351807}}};
		const float b_layer_5[4] = {0.27153488993644714,0.7057781219482422,0.17182892560958862,-0.1903594434261322};

		// Fill with biases for layer_5
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_5[i];
			}
		}

		// Apply main filter for layer_5
		// x_odd[:,248:] = sum(w[k]@x_even[:,248-(0-k)*0:L-(0-k)*0] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-248, 2, 1.0, &w_layer_5[k][0][0], 4, &x_even[248-offset][0], MAX_CH, 1.0, &x_odd[248][0], MAX_CH);
		}


		// Leaky Rectified Linear Unit (ReLU)
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = v > 0 ? v : 0.01f*v;
			}
		}


		// auto-generated code for layer layer_6: Conv1d(4, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_6[1][4][4] = {{{0.46878793835639954,-1.0387492179870605,-0.5741215348243713,-0.522189736366272},{-0.19420407712459564,0.43656766414642334,0.6049628257751465,0.08484707772731781},{0.7904306650161743,-0.4715190529823303,-0.559847891330719,0.02811579406261444},{-0.8615669012069702,0.23395943641662598,0.013578529469668865,-0.1975146234035492}}};
		const float b_layer_6[4] = {0.29546502232551575,0.39104390144348145,-0.10135776549577713,-0.1415109634399414};

		// Fill with biases for layer_6
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_even[l][i] = b_layer_6[i];
			}
		}

		// Apply main filter for layer_6
		// x_even[:,248:] = sum(w[k]@x_odd[:,248-(0-k)*0:L-(0-k)*0] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-248, 4, 1.0, &w_layer_6[k][0][0], 4, &x_odd[248-offset][0], MAX_CH, 1.0, &x_even[248][0], MAX_CH);
		}


		// Leaky Rectified Linear Unit (ReLU)
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = v > 0 ? v : 0.01f*v;
			}
		}


		// auto-generated code for layer layer_7: Conv1d(4, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_7[1][4][4] = {{{-0.2930261194705963,-0.5371442437171936,-1.1685259342193604,-0.10942050814628601},{0.8251791596412659,0.6320266127586365,-0.13638047873973846,-0.2358490377664566},{0.07281064242124557,0.31584805250167847,0.7620288133621216,-0.6519749164581299},{0.03293425217270851,0.17876243591308594,0.18336407840251923,0.2366502434015274}}};
		const float b_layer_7[4] = {0.22507676482200623,0.18497733771800995,-0.4604455530643463,0.4811556041240692};

		// Fill with biases for layer_7
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				x_odd[l][i] = b_layer_7[i];
			}
		}

		// Apply main filter for layer_7
		// x_odd[:,248:] = sum(w[k]@x_even[:,248-(0-k)*0:L-(0-k)*0] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, L-248, 4, 1.0, &w_layer_7[k][0][0], 4, &x_even[248-offset][0], MAX_CH, 1.0, &x_odd[248][0], MAX_CH);
		}


		// Leaky Rectified Linear Unit (ReLU)
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 4; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = v > 0 ? v : 0.01f*v;
			}
		}


		// auto-generated code for layer layer_8: Conv1d(4, 2, kernel_size=(1,), stride=(1,))
		const float w_layer_8[1][4][2] = {{{0.24942860007286072,0.47676345705986023},{-0.2578943073749542,0.7017513513565063},{0.2840387523174286,0.5464609265327454},{0.8155749440193176,-0.3006892204284668}}};
		const float b_layer_8[2] = {-0.21998699009418488,-0.35214468836784363};

		// Fill with biases for layer_8
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 2; i++) {
				x_even[l][i] = b_layer_8[i];
			}
		}

		// Apply main filter for layer_8
		// x_even[:,248:] = sum(w[k]@x_odd[:,248-(0-k)*0:L-(0-k)*0] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-248, 4, 1.0, &w_layer_8[k][0][0], 2, &x_odd[248-offset][0], MAX_CH, 1.0, &x_even[248][0], MAX_CH);
		}


		// Hard Tanh (i.e. hard clip)
		for (int l = 248; l < L; l++) {
			for (int i = 0; i < 2; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = v > 1 ? 1 : v < -1 ? -1 : v;
			}
		}


		// auto-generated code for layer layer_9: Conv1d(2, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_9[2][2][3] = {{{0.3457864820957184,0.2017006278038025,-0.08579321205615997},{-0.47515544295310974,0.11036720126867294,-0.5713734030723572}},{{0.39597341418266296,-0.588678240776062,-0.010333016514778137},{0.4554772973060608,-0.44205543398857117,-0.11183140426874161}}};

		// Apply main filter for layer_9
		// x_odd[:,312:] = sum(w[k]@x_even[:,312-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-312, 2, 1.0, &w_layer_9[k][0][0], 3, &x_even[312-offset][0], MAX_CH, beta, &x_odd[312][0], MAX_CH);
		}


		// auto-generated code for layer layer_10: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_10[2][3][3] = {{{0.06668511033058167,0.09095953404903412,0.3919820785522461},{-0.35606080293655396,-0.0413484200835228,-0.39697495102882385},{-0.19380693137645721,-0.07435861229896545,0.2395181804895401}},{{-0.0011311719426885247,0.34549230337142944,0.37362122535705566},{0.17972669005393982,0.28625473380088806,0.1842804104089737},{-0.0405285619199276,0.529047966003418,-0.1694999486207962}}};

		// Apply main filter for layer_10
		// x_even[:,344:] = sum(w[k]@x_odd[:,344-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-344, 3, 1.0, &w_layer_10[k][0][0], 3, &x_odd[344-offset][0], MAX_CH, beta, &x_even[344][0], MAX_CH);
		}


		// auto-generated code for layer layer_11: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_11[2][3][3] = {{{0.236443430185318,-0.1840898096561432,0.2735786437988281},{-0.13101382553577423,0.06073698028922081,0.015715980902314186},{-0.20564328134059906,0.030758032575249672,0.5833858847618103}},{{-0.006452226545661688,0.18501423299312592,-0.2180073857307434},{-0.3359214961528778,-0.7389189600944519,-0.08783022314310074},{0.19907039403915405,-0.17259112000465393,0.173727884888649}}};

		// Apply main filter for layer_11
		// x_odd[:,360:] = sum(w[k]@x_even[:,360-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-360, 3, 1.0, &w_layer_11[k][0][0], 3, &x_even[360-offset][0], MAX_CH, beta, &x_odd[360][0], MAX_CH);
		}


		// auto-generated code for layer layer_12: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_12[2][3][3] = {{{-0.002768596401438117,0.1874304860830307,-0.17161692678928375},{0.22288878262043,0.16477134823799133,-0.2583368420600891},{-0.3156088888645172,0.19047212600708008,0.6013281345367432}},{{-0.3924002945423126,-0.5904101729393005,-0.03686947748064995},{-0.34470826387405396,-1.0326542854309082,-0.09133326262235641},{0.11502721160650253,-0.30480262637138367,0.16050338745117188}}};

		// Apply main filter for layer_12
		// x_even[:,368:] = sum(w[k]@x_odd[:,368-(1-k)*8:L-(1-k)*8] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*8;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-368, 3, 1.0, &w_layer_12[k][0][0], 3, &x_odd[368-offset][0], MAX_CH, beta, &x_even[368][0], MAX_CH);
		}


		// auto-generated code for layer layer_13: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(8,), bias=False)
		const float w_layer_13[2][3][3] = {{{0.06154458597302437,0.4277319610118866,-0.19913718104362488},{0.2320869266986847,0.10846192389726639,-0.3563246428966522},{0.2108086347579956,-0.47246477007865906,-0.5290394425392151}},{{0.3458643853664398,-0.16748254001140594,0.2945428192615509},{0.6348763108253479,-0.3834710419178009,0.4309787154197693},{-0.1563127040863037,-0.2597964107990265,0.13127759099006653}}};

		// Apply main filter for layer_13
		// x_odd[:,372:] = sum(w[k]@x_even[:,372-(1-k)*4:L-(1-k)*4] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*4;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-372, 3, 1.0, &w_layer_13[k][0][0], 3, &x_even[372-offset][0], MAX_CH, beta, &x_odd[372][0], MAX_CH);
		}


		// auto-generated code for layer layer_14: Conv1d(3, 1, kernel_size=(2,), stride=(1,), dilation=(4,))
		const float w_layer_14[2][3][1] = {{{0.0833425223827362},{-0.4315895438194275},{-0.37653854489326477}},{{-0.46526649594306946},{0.20528174936771393},{-0.41660594940185547}}};
		const float b_layer_14[1] = {0.0004577063082251698};

		// Fill with biases for layer_14
		for (int l = 374; l < L; l++) {
			for (int i = 0; i < 1; i++) {
				x_even[l][i] = b_layer_14[i];
			}
		}

		// Apply main filter for layer_14
		// x_even[:,374:] = sum(w[k]@x_odd[:,374-(1-k)*2:L-(1-k)*2] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*2;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, L-374, 3, 1.0, &w_layer_14[k][0][0], 1, &x_odd[374-offset][0], MAX_CH, 1.0, &x_even[374][0], MAX_CH);
		}


		// Copy result back to y
		for (int l = 374; l < L; l++) {
			y[l] = x_even[l][0];
		}
	}
};
