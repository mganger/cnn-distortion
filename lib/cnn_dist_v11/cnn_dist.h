
#pragma once
extern "C" {
#include <cblas.h>
};

#include <cmath>

struct cnn_dist_v11 {
	const static int latency = 988;
	const static int MAX_L = MAX_BUFFER + latency;
	const static int MAX_CH = 3;
	// About 6e-06*(MAX_BUFFER+988) MB of buffer
	float x_even[MAX_L][MAX_CH];
	float x_odd [MAX_L][MAX_CH];

	void operator()(float* x, float* y, int L) {

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {
			x_odd[i][0] = x[i];
		}

		// auto-generated code for layer layer_0: Conv1d(1, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_0[2][1][3] = {{{0.5425165891647339,-0.7916250824928284,0.9700367450714111}},{{0.397832453250885,0.6385343074798584,0.01968999020755291}}};

		// Apply main filter for layer_0
		// x_even[:,128:] = sum(w[k]@x_odd[:,128-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-128, 1, 1.0, &w_layer_0[k][0][0], 3, &x_odd[128-offset][0], MAX_CH, beta, &x_even[128][0], MAX_CH);
		}


		// auto-generated code for layer layer_1: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_1[2][3][3] = {{{0.08171261847019196,-0.01594649814069271,0.11193780601024628},{0.4005199372768402,-0.5766727924346924,-0.12704743444919586},{-0.2588422894477844,-0.2761650085449219,-0.3044944405555725}},{{-0.6503967046737671,0.322218656539917,-0.5797004699707031},{0.5026640295982361,-0.4251786768436432,0.3183766007423401},{-0.7663193345069885,0.340419203042984,-0.7935641407966614}}};

		// Apply main filter for layer_1
		// x_odd[:,192:] = sum(w[k]@x_even[:,192-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-192, 3, 1.0, &w_layer_1[k][0][0], 3, &x_even[192-offset][0], MAX_CH, beta, &x_odd[192][0], MAX_CH);
		}


		// auto-generated code for layer layer_2: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_2[2][3][3] = {{{0.2756444215774536,-0.19088910520076752,0.14915089309215546},{-0.48789355158805847,0.5590568780899048,-0.2283748984336853},{0.08091071248054504,0.13832536339759827,0.009367907419800758}},{{0.7276689410209656,-0.3082464039325714,0.38449910283088684},{-0.4806201457977295,-0.04889483004808426,-0.0550905279815197},{0.22392182052135468,-0.7621288895606995,0.7460988759994507}}};

		// Apply main filter for layer_2
		// x_even[:,224:] = sum(w[k]@x_odd[:,224-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-224, 3, 1.0, &w_layer_2[k][0][0], 3, &x_odd[224-offset][0], MAX_CH, beta, &x_even[224][0], MAX_CH);
		}


		// auto-generated code for layer layer_3: Conv1d(3, 2, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_3[2][3][2] = {{{-0.30540862679481506,0.1550384908914566},{-0.07137426733970642,0.33228063583374023},{-0.0894075259566307,-0.058681413531303406}},{{-0.007179901469498873,-0.7423701882362366},{-0.5469716787338257,0.4396144449710846},{0.6574147939682007,-0.27030813694000244}}};

		// Apply main filter for layer_3
		// x_odd[:,240:] = sum(w[k]@x_even[:,240-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-240, 3, 1.0, &w_layer_3[k][0][0], 2, &x_even[240-offset][0], MAX_CH, beta, &x_odd[240][0], MAX_CH);
		}


		// auto-generated code for layer layer_4: Conv1d(2, 3, kernel_size=(1,), stride=(1,))
		const float w_layer_4[1][2][3] = {{{-0.46531903743743896,0.5639012455940247,-0.3235679566860199},{-0.6056084036827087,-0.627588152885437,0.5505780577659607}}};
		const float b_layer_4[3] = {0.04911971092224121,0.5733181238174438,-0.44843003153800964};

		// Fill with biases for layer_4
		for (int l = 240; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				x_even[l][i] = b_layer_4[i];
			}
		}

		// Apply main filter for layer_4
		// x_even[:,240:] = sum(w[k]@x_odd[:,240-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-240, 2, 1.0, &w_layer_4[k][0][0], 3, &x_odd[240-offset][0], MAX_CH, 1.0, &x_even[240][0], MAX_CH);
		}


		// Rectified Linear Unit (ReLU)
		for (int l = 240; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = v > 0 ? v : 0;
			}
		}


		// auto-generated code for layer layer_5: Conv1d(3, 3, kernel_size=(1,), stride=(1,))
		const float w_layer_5[1][3][3] = {{{0.31687450408935547,0.2762996256351471,0.4435521364212036},{-0.031989049166440964,0.5671321749687195,-0.5305595397949219},{-0.4244208335876465,-0.2088633030653,-0.19336426258087158}}};
		const float b_layer_5[3] = {0.3715111017227173,0.18572832643985748,-0.5536786913871765};

		// Fill with biases for layer_5
		for (int l = 240; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				x_odd[l][i] = b_layer_5[i];
			}
		}

		// Apply main filter for layer_5
		// x_odd[:,240:] = sum(w[k]@x_even[:,240-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-240, 3, 1.0, &w_layer_5[k][0][0], 3, &x_even[240-offset][0], MAX_CH, 1.0, &x_odd[240][0], MAX_CH);
		}


		// Rectified Linear Unit (ReLU)
		for (int l = 240; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = v > 0 ? v : 0;
			}
		}


		// auto-generated code for layer layer_6: Conv1d(3, 2, kernel_size=(1,), stride=(1,))
		const float w_layer_6[1][3][2] = {{{-0.6973801255226135,0.482199490070343},{-0.5523191094398499,0.7169191241264343},{-0.028134644031524658,0.47927606105804443}}};
		const float b_layer_6[2] = {0.14842456579208374,-0.7049865126609802};

		// Fill with biases for layer_6
		for (int l = 240; l < L; l++) {
			for (int i = 0; i < 2; i++) {
				x_even[l][i] = b_layer_6[i];
			}
		}

		// Apply main filter for layer_6
		// x_even[:,240:] = sum(w[k]@x_odd[:,240-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-240, 3, 1.0, &w_layer_6[k][0][0], 2, &x_odd[240-offset][0], MAX_CH, 1.0, &x_even[240][0], MAX_CH);
		}


		// auto-generated code for layer layer_7: Conv1d(2, 2, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_7[2][2][2] = {{{0.6801486015319824,0.619464635848999},{-0.6988365650177002,-1.0319504737854004}},{{-0.5953407883644104,-0.03890533372759819},{-0.04774300754070282,-0.23442652821540833}}};

		// Apply main filter for layer_7
		// x_odd[:,368:] = sum(w[k]@x_even[:,368-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-368, 2, 1.0, &w_layer_7[k][0][0], 2, &x_even[368-offset][0], MAX_CH, beta, &x_odd[368][0], MAX_CH);
		}


		// auto-generated code for layer layer_8: Conv1d(2, 2, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_8[2][2][2] = {{{-0.00916073564440012,-0.3244231343269348},{0.3850797712802887,-0.35271331667900085}},{{0.864255428314209,0.437042236328125},{0.8403602242469788,1.5695102214813232}}};

		// Apply main filter for layer_8
		// x_even[:,432:] = sum(w[k]@x_odd[:,432-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-432, 2, 1.0, &w_layer_8[k][0][0], 2, &x_odd[432-offset][0], MAX_CH, beta, &x_even[432][0], MAX_CH);
		}


		// auto-generated code for layer layer_9: Conv1d(2, 2, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_9[2][2][2] = {{{0.16739481687545776,-0.6928492188453674},{0.29013216495513916,-0.44314369559288025}},{{-0.4475366473197937,0.18571852147579193},{-0.7497075796127319,0.40954190492630005}}};

		// Apply main filter for layer_9
		// x_odd[:,464:] = sum(w[k]@x_even[:,464-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-464, 2, 1.0, &w_layer_9[k][0][0], 2, &x_even[464-offset][0], MAX_CH, beta, &x_odd[464][0], MAX_CH);
		}


		// auto-generated code for layer layer_10: Conv1d(2, 2, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_10[2][2][2] = {{{0.6991646885871887,1.3733302354812622},{-0.5246726870536804,-1.1175775527954102}},{{-0.30855458974838257,-0.16230124235153198},{-0.6418540477752686,-0.7104082107543945}}};

		// Apply main filter for layer_10
		// x_even[:,480:] = sum(w[k]@x_odd[:,480-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-480, 2, 1.0, &w_layer_10[k][0][0], 2, &x_odd[480-offset][0], MAX_CH, beta, &x_even[480][0], MAX_CH);
		}


		// auto-generated code for layer layer_11: Conv1d(2, 3, kernel_size=(1,), stride=(1,))
		const float w_layer_11[1][2][3] = {{{0.35057613253593445,0.19413475692272186,-1.7766963243484497},{0.7230758666992188,0.8826907873153687,-1.143256425857544}}};
		const float b_layer_11[3] = {0.3755781948566437,-0.04831961169838905,0.17314332723617554};

		// Fill with biases for layer_11
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				x_odd[l][i] = b_layer_11[i];
			}
		}

		// Apply main filter for layer_11
		// x_odd[:,480:] = sum(w[k]@x_even[:,480-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-480, 2, 1.0, &w_layer_11[k][0][0], 3, &x_even[480-offset][0], MAX_CH, 1.0, &x_odd[480][0], MAX_CH);
		}


		// Rectified Linear Unit (ReLU)
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				auto& v = x_odd[i][l];
				x_odd[l][i] = v > 0 ? v : 0;
			}
		}


		// auto-generated code for layer layer_12: Conv1d(3, 3, kernel_size=(1,), stride=(1,))
		const float w_layer_12[1][3][3] = {{{0.22386933863162994,0.3896096646785736,-1.2931337356567383},{-0.3069695234298706,-0.20638851821422577,0.21180398762226105},{-0.9692084789276123,-0.7691777944564819,0.014784413389861584}}};
		const float b_layer_12[3] = {0.4985290765762329,0.14306029677391052,0.6151387095451355};

		// Fill with biases for layer_12
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				x_even[l][i] = b_layer_12[i];
			}
		}

		// Apply main filter for layer_12
		// x_even[:,480:] = sum(w[k]@x_odd[:,480-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-480, 3, 1.0, &w_layer_12[k][0][0], 3, &x_odd[480-offset][0], MAX_CH, 1.0, &x_even[480][0], MAX_CH);
		}


		// Rectified Linear Unit (ReLU)
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 3; i++) {
				auto& v = x_even[i][l];
				x_even[l][i] = v > 0 ? v : 0;
			}
		}


		// auto-generated code for layer layer_13: Conv1d(3, 2, kernel_size=(1,), stride=(1,))
		const float w_layer_13[1][3][2] = {{{-0.593827486038208,-0.38016748428344727},{-0.4597407281398773,-0.33571743965148926},{0.7453767657279968,0.19495870172977448}}};
		const float b_layer_13[2] = {0.22208648920059204,0.16918933391571045};

		// Fill with biases for layer_13
		for (int l = 480; l < L; l++) {
			for (int i = 0; i < 2; i++) {
				x_odd[l][i] = b_layer_13[i];
			}
		}

		// Apply main filter for layer_13
		// x_odd[:,480:] = sum(w[k]@x_even[:,480-(0-k)*1:L-(0-k)*1] for k in w.shape[0])
		for (int k = 0; k < 1; k++) {
			int offset = (0-k)*1;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, L-480, 3, 1.0, &w_layer_13[k][0][0], 2, &x_even[480-offset][0], MAX_CH, 1.0, &x_odd[480][0], MAX_CH);
		}


		// auto-generated code for layer layer_14: Conv1d(2, 3, kernel_size=(2,), stride=(1,), dilation=(256,), bias=False)
		const float w_layer_14[2][2][3] = {{{-0.2930094599723816,-0.3145929276943207,-0.001693510334007442},{0.1197657510638237,-0.13549013435840607,0.07274822890758514}},{{-0.4933432340621948,0.5659435987472534,0.19843052327632904},{-0.27480918169021606,0.35560739040374756,0.10985294729471207}}};

		// Apply main filter for layer_14
		// x_even[:,736:] = sum(w[k]@x_odd[:,736-(1-k)*256:L-(1-k)*256] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*256;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-736, 2, 1.0, &w_layer_14[k][0][0], 3, &x_odd[736-offset][0], MAX_CH, beta, &x_even[736][0], MAX_CH);
		}


		// auto-generated code for layer layer_15: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(128,), bias=False)
		const float w_layer_15[2][3][3] = {{{0.21634314954280853,-0.00676488783210516,0.23325806856155396},{-0.279815137386322,-0.33767443895339966,-0.30186694860458374},{-0.03569157421588898,-0.03750299662351608,-0.07475467771291733}},{{0.4385875165462494,-0.18668220937252045,-0.5759834051132202},{-0.3424713909626007,-0.2807602286338806,0.43463683128356934},{-0.14124943315982819,0.12618812918663025,0.45685380697250366}}};

		// Apply main filter for layer_15
		// x_odd[:,864:] = sum(w[k]@x_even[:,864-(1-k)*128:L-(1-k)*128] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*128;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-864, 3, 1.0, &w_layer_15[k][0][0], 3, &x_even[864-offset][0], MAX_CH, beta, &x_odd[864][0], MAX_CH);
		}


		// auto-generated code for layer layer_16: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(64,), bias=False)
		const float w_layer_16[2][3][3] = {{{-0.06554117053747177,0.003141622059047222,-0.051216039806604385},{-0.4536784589290619,0.25759357213974,-0.3698713481426239},{0.13542863726615906,-0.2283167690038681,0.22306053340435028}},{{-0.3607301414012909,0.37811559438705444,0.5017681121826172},{-0.22112470865249634,0.10664751380681992,-0.29574689269065857},{-0.21897876262664795,-0.32229962944984436,-0.5484582185745239}}};

		// Apply main filter for layer_16
		// x_even[:,928:] = sum(w[k]@x_odd[:,928-(1-k)*64:L-(1-k)*64] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*64;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-928, 3, 1.0, &w_layer_16[k][0][0], 3, &x_odd[928-offset][0], MAX_CH, beta, &x_even[928][0], MAX_CH);
		}


		// auto-generated code for layer layer_17: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(32,), bias=False)
		const float w_layer_17[2][3][3] = {{{0.29007214307785034,-0.0548742339015007,-0.5267331004142761},{-0.4545048475265503,-0.07398826628923416,0.3953342139720917},{0.3166656792163849,0.3342866897583008,0.176069974899292}},{{0.0034685502760112286,-0.19395452737808228,-0.4881903827190399},{0.2198597639799118,-0.5299757122993469,0.035136885941028595},{0.41146162152290344,-0.3213978111743927,0.27530133724212646}}};

		// Apply main filter for layer_17
		// x_odd[:,960:] = sum(w[k]@x_even[:,960-(1-k)*32:L-(1-k)*32] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*32;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-960, 3, 1.0, &w_layer_17[k][0][0], 3, &x_even[960-offset][0], MAX_CH, beta, &x_odd[960][0], MAX_CH);
		}


		// auto-generated code for layer layer_18: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(16,), bias=False)
		const float w_layer_18[2][3][3] = {{{0.3139098286628723,0.16119544208049774,-0.572413980960846},{-0.02275865525007248,-0.7924837470054626,0.4925481975078583},{0.20211368799209595,0.433534175157547,-0.19771604239940643}},{{0.08686484396457672,-0.1644708812236786,-0.07797105610370636},{0.44291236996650696,-0.0879431813955307,0.22097747027873993},{0.0974498838186264,0.17221441864967346,0.4408397674560547}}};

		// Apply main filter for layer_18
		// x_even[:,976:] = sum(w[k]@x_odd[:,976-(1-k)*16:L-(1-k)*16] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*16;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-976, 3, 1.0, &w_layer_18[k][0][0], 3, &x_odd[976-offset][0], MAX_CH, beta, &x_even[976][0], MAX_CH);
		}


		// auto-generated code for layer layer_19: Conv1d(3, 3, kernel_size=(2,), stride=(1,), dilation=(8,), bias=False)
		const float w_layer_19[2][3][3] = {{{0.3545910716056824,0.03991038724780083,0.04608919844031334},{0.21452970802783966,0.040388401597738266,-0.34851768612861633},{0.016625389456748962,-0.187050461769104,-0.20521129667758942}},{{-0.0030056943651288748,-0.16199806332588196,0.3153594434261322},{0.021018516272306442,-0.2926247715950012,0.24942223727703094},{0.35363298654556274,0.5592989921569824,-0.5676087737083435}}};

		// Apply main filter for layer_19
		// x_odd[:,984:] = sum(w[k]@x_even[:,984-(1-k)*8:L-(1-k)*8] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*8;
			float beta = k == 0 ? 0.0 : 1.0;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, L-984, 3, 1.0, &w_layer_19[k][0][0], 3, &x_even[984-offset][0], MAX_CH, beta, &x_odd[984][0], MAX_CH);
		}


		// auto-generated code for layer layer_20: Conv1d(3, 1, kernel_size=(2,), stride=(1,), dilation=(4,))
		const float w_layer_20[2][3][1] = {{{0.08755119144916534},{0.10478556156158447},{0.0692603662610054}},{{0.49552807211875916},{0.3751060366630554},{-0.5716832280158997}}};
		const float b_layer_20[1] = {0.0005086741293780506};

		// Fill with biases for layer_20
		for (int l = 988; l < L; l++) {
			for (int i = 0; i < 1; i++) {
				x_even[l][i] = b_layer_20[i];
			}
		}

		// Apply main filter for layer_20
		// x_even[:,988:] = sum(w[k]@x_odd[:,988-(1-k)*4:L-(1-k)*4] for k in w.shape[0])
		for (int k = 0; k < 2; k++) {
			int offset = (1-k)*4;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, L-988, 3, 1.0, &w_layer_20[k][0][0], 1, &x_odd[988-offset][0], MAX_CH, 1.0, &x_even[988][0], MAX_CH);
		}


		// Copy result back to y
		for (int l = 988; l < L; l++) {
			y[l] = x_even[l][0];
		}
	}
};
