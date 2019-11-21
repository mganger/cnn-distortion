
extern "C" {
#include <cblas.h>
};

#include <cmath>

struct cnn_dist_v12 {
	const static int latency = 748;
	const static int MAX_L = MAX_BUFFER + latency;
	// About 8e-06*(MAX_BUFFER+748) MB of buffer
	float x_even[4][MAX_L];
	float x_odd [4][MAX_L];

	void operator()(float* x, float* y, int L) {

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {
			x_odd[0][i] = x[i];
		}

		// auto-generated code for layer layer_0: Conv1d(1, 3, kernel_size=(2,), stride=(1,), dilation=(256,), bias=False)
		const float w_layer_0[2][3][1] = {{{-0.08583030849695206},{0.5186648368835449},{-0.8003621101379395}},{{-1.0631130933761597},{1.0840036869049072},{0.8071127533912659}}};

		// Apply main filter for layer_0
		// x_even[:,256:] = sum(w[k]@x_odd[:,256-(1-k)*256:L-(1-k)*256] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*256;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-256, 1, 1.0, &w_layer_0[k][0][0], 1, &x_odd[0][256-offset], MAX_L, k==0?0.0:1.0, &x_even[0][256], MAX_L);
		}


		// auto-generated code for layer layer_1: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_1[2][3][3] = {{{0.2636786699295044,0.010127183049917221,-0.7445932030677795},{1.0707275867462158,-1.2835906744003296,-0.7622990608215332},{-0.28979936242103577,0.24459198117256165,-0.005548056680709124}},{{-0.03021678701043129,-0.28591257333755493,-0.031202159821987152},{-0.2614839971065521,-0.26638010144233704,0.04599398747086525},{0.4230376183986664,-0.26542794704437256,0.7870035767555237}}};

		// Apply main filter for layer_1
		// x_odd[:,384:] = sum(w[k]@x_even[:,384-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-384, 3, 1.0, &w_layer_1[k][0][0], 3, &x_even[0][384-offset], MAX_L, k==0?0.0:1.0, &x_odd[0][384], MAX_L);
		}


		// auto-generated code for layer layer_2: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_2[2][3][3] = {{{0.2764991819858551,-0.09734676778316498,-0.18309541046619415},{0.619861364364624,0.6869813799858093,0.029998496174812317},{0.42969435453414917,0.5311681628227234,0.18954531848430634}},{{-0.34401893615722656,-1.0063894987106323,0.16109558939933777},{0.018978215754032135,0.29526472091674805,0.9205557107925415},{-0.10648131370544434,-0.63907790184021,0.6154322028160095}}};

		// Apply main filter for layer_2
		// x_even[:,448:] = sum(w[k]@x_odd[:,448-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-448, 3, 1.0, &w_layer_2[k][0][0], 3, &x_odd[0][448-offset], MAX_L, k==0?0.0:1.0, &x_even[0][448], MAX_L);
		}


		// auto-generated code for layer layer_3: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_3[2][3][3] = {{{0.39770349860191345,-0.8676868081092834,-0.03558063134551048},{-0.7468500733375549,0.19269534945487976,-0.985572099685669},{-0.5976365804672241,0.2044781595468521,-0.4192846119403839}},{{0.2553304135799408,-0.6577630639076233,-0.5886655449867249},{-0.12198462337255478,-0.4945431351661682,-0.10778156667947769},{0.4148417115211487,-0.5318288803100586,-0.44109970331192017}}};

		// Apply main filter for layer_3
		// x_odd[:,480:] = sum(w[k]@x_even[:,480-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-480, 3, 1.0, &w_layer_3[k][0][0], 3, &x_even[0][480-offset], MAX_L, k==0?0.0:1.0, &x_odd[0][480], MAX_L);
		}


		// auto-generated code for layer layer_4: Conv1d(3, 2, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_4[2][2][3] = {{{-0.6477870345115662,-0.10562854260206223,-0.2741983234882355},{-0.610987663269043,0.5860682129859924,0.2380448579788208}},{{0.4446178674697876,-0.9076033234596252,-0.7311594486236572},{-0.04853849858045578,-1.0434296131134033,-0.7338244915008545}}};

		// Apply main filter for layer_4
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, L-496, 3, 1.0, &w_layer_4[k][0][0], 3, &x_odd[0][496-offset], MAX_L, k==0?0.0:1.0, &x_even[0][496], MAX_L);
		}


		// auto-generated code for layer layer_5: Conv1d(2, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_5[1][4][2] = {{{-0.5009680390357971,-1.5255439281463623},{0.8821141719818115,0.39772993326187134},{-1.702233076095581,-0.1289607584476471},{0.47020605206489563,0.9674017429351807}}};
		const float b_layer_5[4] = {0.27153488993644714,0.7057781219482422,0.17182892560958862,-0.1903594434261322};

		// Fill with biases for layer_5
		for (int i = 0; i < 4; i++) {
			for (int l = 496; l < L; l++) {
				x_odd[i][l] = b_layer_5[i];
			}
		}

		// Apply main filter for layer_5
		// x_odd[:,496:] = sum(w[k]@x_even[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 2, 1.0, &w_layer_5[k][0][0], 2, &x_even[0][496-offset], MAX_L, 1.0, &x_odd[0][496], MAX_L);
		}


		// Leaky Rectified Linear Unit (ReLU)
		for (int i = 0; i < 4; i++) {
			for (int l = 496; l < L; l++) {
				x_odd[i][l] = x_odd[i][l] > 0 ? x_odd[i][l] : 0.01f*x_odd[i][l];
			}
		}


		// auto-generated code for layer layer_6: Conv1d(4, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_6[1][4][4] = {{{0.46878793835639954,-0.19420407712459564,0.7904306650161743,-0.8615669012069702},{-1.0387492179870605,0.43656766414642334,-0.4715190529823303,0.23395943641662598},{-0.5741215348243713,0.6049628257751465,-0.559847891330719,0.013578529469668865},{-0.522189736366272,0.08484707772731781,0.02811579406261444,-0.1975146234035492}}};
		const float b_layer_6[4] = {0.29546502232551575,0.39104390144348145,-0.10135776549577713,-0.1415109634399414};

		// Fill with biases for layer_6
		for (int i = 0; i < 4; i++) {
			for (int l = 496; l < L; l++) {
				x_even[i][l] = b_layer_6[i];
			}
		}

		// Apply main filter for layer_6
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 4, 1.0, &w_layer_6[k][0][0], 4, &x_odd[0][496-offset], MAX_L, 1.0, &x_even[0][496], MAX_L);
		}


		// Leaky Rectified Linear Unit (ReLU)
		for (int i = 0; i < 4; i++) {
			for (int l = 496; l < L; l++) {
				x_even[i][l] = x_even[i][l] > 0 ? x_even[i][l] : 0.01f*x_even[i][l];
			}
		}


		// auto-generated code for layer layer_7: Conv1d(4, 4, kernel_size=(1,), stride=(1,))
		const float w_layer_7[1][4][4] = {{{-0.2930261194705963,0.8251791596412659,0.07281064242124557,0.03293425217270851},{-0.5371442437171936,0.6320266127586365,0.31584805250167847,0.17876243591308594},{-1.1685259342193604,-0.13638047873973846,0.7620288133621216,0.18336407840251923},{-0.10942050814628601,-0.2358490377664566,-0.6519749164581299,0.2366502434015274}}};
		const float b_layer_7[4] = {0.22507676482200623,0.18497733771800995,-0.4604455530643463,0.4811556041240692};

		// Fill with biases for layer_7
		for (int i = 0; i < 4; i++) {
			for (int l = 496; l < L; l++) {
				x_odd[i][l] = b_layer_7[i];
			}
		}

		// Apply main filter for layer_7
		// x_odd[:,496:] = sum(w[k]@x_even[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, L-496, 4, 1.0, &w_layer_7[k][0][0], 4, &x_even[0][496-offset], MAX_L, 1.0, &x_odd[0][496], MAX_L);
		}


		// Leaky Rectified Linear Unit (ReLU)
		for (int i = 0; i < 4; i++) {
			for (int l = 496; l < L; l++) {
				x_odd[i][l] = x_odd[i][l] > 0 ? x_odd[i][l] : 0.01f*x_odd[i][l];
			}
		}


		// auto-generated code for layer layer_8: Conv1d(4, 2, kernel_size=(1,), stride=(1,))
		const float w_layer_8[1][2][4] = {{{0.24942860007286072,-0.2578943073749542,0.2840387523174286,0.8155749440193176},{0.47676345705986023,0.7017513513565063,0.5464609265327454,-0.3006892204284668}}};
		const float b_layer_8[2] = {-0.21998699009418488,-0.35214468836784363};

		// Fill with biases for layer_8
		for (int i = 0; i < 2; i++) {
			for (int l = 496; l < L; l++) {
				x_even[i][l] = b_layer_8[i];
			}
		}

		// Apply main filter for layer_8
		// x_even[:,496:] = sum(w[k]@x_odd[:,496-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, L-496, 4, 1.0, &w_layer_8[k][0][0], 4, &x_odd[0][496-offset], MAX_L, 1.0, &x_even[0][496], MAX_L);
		}


		// Hard Tanh (i.e. hard clip)
		for (int i = 0; i < 2; i++) {
			for (int l = 496; l < L; l++) {
				auto v = x_even[i][l];
				x_even[i][l] = v > 1 ? 1 : v < -1 ? -1 : v;
			}
		}


		// auto-generated code for layer layer_9: Conv1d(2, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_9[2][3][2] = {{{0.3457864820957184,-0.47515544295310974},{0.2017006278038025,0.11036720126867294},{-0.08579321205615997,-0.5713734030723572}},{{0.39597341418266296,0.4554772973060608},{-0.588678240776062,-0.44205543398857117},{-0.010333016514778137,-0.11183140426874161}}};

		// Apply main filter for layer_9
		// x_odd[:,624:] = sum(w[k]@x_even[:,624-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-624, 2, 1.0, &w_layer_9[k][0][0], 2, &x_even[0][624-offset], MAX_L, k==0?0.0:1.0, &x_odd[0][624], MAX_L);
		}


		// auto-generated code for layer layer_10: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_10[2][3][3] = {{{0.06668511033058167,-0.35606080293655396,-0.19380693137645721},{0.09095953404903412,-0.0413484200835228,-0.07435861229896545},{0.3919820785522461,-0.39697495102882385,0.2395181804895401}},{{-0.0011311719426885247,0.17972669005393982,-0.0405285619199276},{0.34549230337142944,0.28625473380088806,0.529047966003418},{0.37362122535705566,0.1842804104089737,-0.1694999486207962}}};

		// Apply main filter for layer_10
		// x_even[:,688:] = sum(w[k]@x_odd[:,688-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-688, 3, 1.0, &w_layer_10[k][0][0], 3, &x_odd[0][688-offset], MAX_L, k==0?0.0:1.0, &x_even[0][688], MAX_L);
		}


		// auto-generated code for layer layer_11: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_11[2][3][3] = {{{0.236443430185318,-0.13101382553577423,-0.20564328134059906},{-0.1840898096561432,0.06073698028922081,0.030758032575249672},{0.2735786437988281,0.015715980902314186,0.5833858847618103}},{{-0.006452226545661688,-0.3359214961528778,0.19907039403915405},{0.18501423299312592,-0.7389189600944519,-0.17259112000465393},{-0.2180073857307434,-0.08783022314310074,0.173727884888649}}};

		// Apply main filter for layer_11
		// x_odd[:,720:] = sum(w[k]@x_even[:,720-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-720, 3, 1.0, &w_layer_11[k][0][0], 3, &x_even[0][720-offset], MAX_L, k==0?0.0:1.0, &x_odd[0][720], MAX_L);
		}


		// auto-generated code for layer layer_12: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_12[2][3][3] = {{{-0.002768596401438117,0.22288878262043,-0.3156088888645172},{0.1874304860830307,0.16477134823799133,0.19047212600708008},{-0.17161692678928375,-0.2583368420600891,0.6013281345367432}},{{-0.3924002945423126,-0.34470826387405396,0.11502721160650253},{-0.5904101729393005,-1.0326542854309082,-0.30480262637138367},{-0.03686947748064995,-0.09133326262235641,0.16050338745117188}}};

		// Apply main filter for layer_12
		// x_even[:,736:] = sum(w[k]@x_odd[:,736-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-736, 3, 1.0, &w_layer_12[k][0][0], 3, &x_odd[0][736-offset], MAX_L, k==0?0.0:1.0, &x_even[0][736], MAX_L);
		}


		// auto-generated code for layer layer_13: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(8,), bias=False)
		const float w_layer_13[2][3][3] = {{{0.06154458597302437,0.2320869266986847,0.2108086347579956},{0.4277319610118866,0.10846192389726639,-0.47246477007865906},{-0.19913718104362488,-0.3563246428966522,-0.5290394425392151}},{{0.3458643853664398,0.6348763108253479,-0.1563127040863037},{-0.16748254001140594,-0.3834710419178009,-0.2597964107990265},{0.2945428192615509,0.4309787154197693,0.13127759099006653}}};

		// Apply main filter for layer_13
		// x_odd[:,744:] = sum(w[k]@x_even[:,744-(1-k)*8:L-(1-k)*8] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*8;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, L-744, 3, 1.0, &w_layer_13[k][0][0], 3, &x_even[0][744-offset], MAX_L, k==0?0.0:1.0, &x_odd[0][744], MAX_L);
		}


		// auto-generated code for layer layer_14: Conv1d(3, 1, kernel_size=(2,), stride=(1,), dilation=(4,))
		const float w_layer_14[2][1][3] = {{{0.0833425223827362,-0.4315895438194275,-0.37653854489326477}},{{-0.46526649594306946,0.20528174936771393,-0.41660594940185547}}};
		const float b_layer_14[1] = {0.0004577063082251698};

		// Fill with biases for layer_14
		for (int i = 0; i < 1; i++) {
			for (int l = 748; l < L; l++) {
				x_even[i][l] = b_layer_14[i];
			}
		}

		// Apply main filter for layer_14
		// x_even[:,748:] = sum(w[k]@x_odd[:,748-(1-k)*4:L-(1-k)*4] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*4;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, L-748, 3, 1.0, &w_layer_14[k][0][0], 3, &x_odd[0][748-offset], MAX_L, 1.0, &x_even[0][748], MAX_L);
		}


		// Copy result back to y
		for (int l = 748; l < L; l++) {
			y[l] = x_even[0][l];
		}
	}
};
