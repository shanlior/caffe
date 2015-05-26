#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"


using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_int32(gpu2, -1,
    "Run in GPU mode on given device ID.");
DEFINE_int32(iteration_distance, 1,
    "The propagation distance of each mapping stage");
DEFINE_int32(start_iteration, 100,
    "The number of iterations before the start of the dual stage");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(solver2, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(solverComb, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(snapshot2, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_string(weights2, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}



void CompareNets(shared_ptr<caffe::Solver<float> > solver1, shared_ptr<caffe::Solver<float> > solver2) {
  const vector<shared_ptr<Blob<float> > >& net_params1 = solver1->net()->params();
  const vector<shared_ptr<Blob<float> > >& net_params2 = solver2->net()->params();
  CHECK(net_params1.size() == net_params2.size()) << "Networks dimensions mismatch";
  int layer_num = 0;
  for (int i = 0; i < net_params1.size(); ++i) {
	double global_contraction_accum_curr = 0.0;
	float delta;
	float w1,w2;
	CHECK(net_params1[i]->count() == net_params2[i]->count()) << "Layer dimensions mismatch";
    int n = net_params1[i]->count();
    // check contraction
    Blob<float> tempVec1;
	tempVec1.Reshape(n,1,1,1);
	w1 = net_params1[i]->sumsq_data();
	w2 = net_params2[i]->sumsq_data();
    switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe::caffe_sub(n,net_params1[i]->cpu_data(),net_params2[i]->cpu_data(),tempVec1.mutable_cpu_data());
        delta = caffe::caffe_cpu_dot(n, tempVec1.cpu_data(), tempVec1.cpu_data());
        break;
      case Caffe::GPU:
        caffe::caffe_gpu_sub(n,net_params1[i]->gpu_data(),net_params2[i]->gpu_data(),tempVec1.mutable_gpu_data());
        caffe::caffe_gpu_dot(n,tempVec1.gpu_data(),tempVec1.gpu_data(),&delta);
        break;
      }
    global_contraction_accum_curr = std::sqrt(delta)/((std::sqrt(w1)+std::sqrt(w2))/2);
    while (!solver1->net()->layers()[layer_num]->blobs().size()) {
      layer_num++;
    }
    if (i % 2 == 0) {
      LOG(INFO) << "Layer #" << i << ": " << solver1->net()->layer_names()[layer_num];
    } else {
      LOG(INFO) << "Layer #" << i << ": bias of " << solver1->net()->layer_names()[layer_num];
      layer_num++;
    }
    LOG(INFO) << "Net parameters count: " << n;
    LOG(INFO) << "Iteration #"<< solver1->iter() <<", Layer #" << i << " Accumulating "<< std::fixed << std::setprecision(10) << global_contraction_accum_curr;
  }
}


float QuadraticEstimate(vector<float> & x,vector<float> & y, float & c0,
		float & c1, float & c2, float & Rsquare) {
  // for c0*x^2 + c1*x + c2

  // auxiliary  calculations
  double x_4 = .0;
  double x_3 = .0;
  double x_2 = .0;
  double x_1 = .0;
  double x_0 = x.size();
  double y_1 = .0;
  double y_x_1 = .0;
  double y_x_2 = .0;
  for (int i = 0; i < x.size(); ++i) {
    x_4 += x[i] * x[i] * x[i] * x[i];
    x_3 += x[i] * x[i] * x[i];
    x_2 += x[i] * x[i];
    x_1 += x[i];
    y_1 += y[i];
    y_x_1 += y[i] * x[i];
    y_x_2 += y[i] * x[i] * x[i];
  }

  c0 = c1 = c2 =.0;
  double det = x_4*(x_2*x_0 - x_1*x_1) - x_3*(x_3*x_0 - x_1*x_2) + x_2*(x_3*x_1 - x_2*x_2);
  double d0 = y_x_2*(x_2*x_0 - x_1*x_1) - y_x_1*(x_3*x_0 - x_1*x_2) + y_1*(x_3*x_1 - x_2*x_2);
  double d1 = x_4*(y_x_1*x_0 - y_1*x_1) - x_3 * (y_x_2*x_0 - y_1*x_2) + x_2*(y_x_2*x_1 - y_x_1*x_2);
  double d2 = x_4*(x_2*y_1 - x_1*y_x_1) - x_3 * (x_3*y_1 - x_1*y_x_2) + x_2*(x_3*y_x_1 - x_2*y_x_2);

  c0 = d0/det;
  c1 = d1/det;
  c2 = d2/det;

  // R2 calculation
  LOG(INFO) << " a = " << c0 << " b = " << c1 << " c = " << c2;
  double error=0.0;
  double distance_from_mean = 0.0;
  double mean = 0.0;
  for (int i = 0; i < x.size(); ++i) {
	mean += y[i];
  }
  mean = mean/x.size();

  for (int i = 0; i < x.size(); ++i) {
	double tmp = y[i] - c0 * x[i] * x[i] - c1 * x[i] - c2;
	double tmp2 = mean - y[i];
    error += tmp*tmp;
	distance_from_mean += tmp2*tmp2;
  }
  Rsquare = (1.0 - error/distance_from_mean);
  LOG(INFO) << "Rsquare = " << Rsquare;
  return (-c1/(2.*c0));
}

float CheckLinearCombination(shared_ptr<caffe::Solver<float> > solver1,
		shared_ptr<caffe::Solver<float> > solver2,
		shared_ptr<caffe::Solver<float> > CombinedSolver,
		float & lastKnownLoss, const float baseLR, const int iteration) {
  // return minimum alpha
  const float step_size = 0.1;
  const float minAlpha = 0.;
  float opt_alpha = minAlpha;
  float opt_loss = 0;
  const float maxAlpha = 1.;
  const vector<shared_ptr<Blob<float> > >& net_params1 = solver1->net()->params();
  const vector<shared_ptr<Blob<float> > >& net_params2 = solver2->net()->params();
  const vector<shared_ptr<Blob<float> > >& net_params = CombinedSolver->net()->params();
  CHECK(net_params1.size() == net_params2.size()) << "Networks dimensions mismatch";
  vector<float> alphas, losses;
  alphas.clear();
  losses.clear();
  LOG(INFO) << "Starting linear combination test iterations";
  float loss_0;
  float loss_1;
  for (float alpha = minAlpha; alpha <= maxAlpha; alpha += step_size) {
    // Part A - make new combined network
	for (int i = 0; i < net_params1.size(); ++i) {
  	  CHECK(net_params1[i]->count() == net_params2[i]->count()) << "Layer dimensions mismatch";
      int n = net_params1[i]->count();
      // check contraction
      Blob<float> tempVec;
      tempVec.Reshape(n,1,1,1);
      switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe::caffe_copy(n,net_params2[i]->cpu_data(), net_params[i]->mutable_cpu_data());
          caffe::caffe_cpu_axpby(n, alpha, net_params1[i]->cpu_data(), (float)(1.0 - alpha), net_params[i]->mutable_cpu_data());
          break;
        case Caffe::GPU:
          caffe::caffe_copy(n,net_params2[i]->gpu_data(), net_params[i]->mutable_gpu_data());
          caffe::caffe_gpu_axpby(n, alpha, net_params1[i]->gpu_data(), (float)(1.0 - alpha), net_params[i]->mutable_gpu_data());
          break;
      }
    }
	// Part B - testLoss and trainLoss
	//float TestLoss = CombinedSolver->TestLoss(alpha); // to check the algorithm we emit the test loss
	float TrainLoss = CombinedSolver->TrainLoss(alpha,10);
	//float SmallTrainLoss = smallSolver->TrainLoss(alpha,50);
	if ((alpha > (0.0 - step_size/2)) && (alpha < (0.0 + step_size/2)))
	  loss_0 = TrainLoss;
	else if (((alpha > (1.0 - step_size/2)) && (alpha < (1.0 + step_size/2))))
	  loss_1 = TrainLoss;

	alphas.push_back(alpha);
	losses.push_back(TrainLoss);
	if (alpha == minAlpha) {
	  opt_loss = TrainLoss;
	} else if (TrainLoss < opt_loss) {
	  opt_loss = TrainLoss;
	  opt_alpha = alpha;
	}
/*
	LOG(INFO) << "Iteration #" << solver1->iter() << " for alpha = " << std::fixed <<
			std::setprecision(2)<< alpha <<
			", Test Loss = " << std::fixed << std::setprecision(8) << TestLoss <<
			" and Train Loss = " << std::fixed << std::setprecision(8) << TrainLoss;
*/

	LOG(INFO) << "Iteration #" << solver1->iter() << " for alpha = " << std::fixed <<
			std::setprecision(2)<< alpha <<
			" and Train Loss = " << std::fixed << std::setprecision(8) << TrainLoss;
/*

	LOG(INFO) << "Iteration #" << solver1->iter() << " for alpha = " << std::fixed <<
				std::setprecision(2)<< alpha <<
				", Test Loss = " << std::fixed << std::setprecision(8) << TestLoss <<
				" and Train Loss = " << std::fixed << std::setprecision(8) << TrainLoss <<
				" and SmallTrain Loss = " << std::fixed << std::setprecision(8) << SmallTrainLoss;
*/

  }
  // Find best fit quadratic function to data
  float c0,c1,c2;
  for (int i=0; i< alphas.size(); i++) {
	  LOG(INFO) << alphas[i] << " " << losses[i];
  }
	LOG(INFO) << "Last known loss: " << lastKnownLoss;
	LOG(INFO) << "Loss_0: " << loss_0;
	LOG(INFO) << "Loss_1: " << loss_1;
/*
  if (lastKnownLoss >= 0.0) {
    if ((lastKnownLoss < loss_0) && (lastKnownLoss < loss_1)) {
	  //const float reduceFactor = 0.9;
	  //const float newLR = reduceFactor*solver1->getBaseLR();
	  const float reduceFactor = baseLR/100;;
	  const float newLR = solver1->getBaseLR() - reduceFactor;
	  if (newLR >= (baseLR/100)) {
        solver1->setBaseLR(newLR);
        solver2->setBaseLR(newLR);
        CombinedSolver->setBaseLR(newLR);
        LOG(INFO) << "Decrease LR";
	    LOG(INFO) << "Current LR is: " << solver1->getBaseLR();
	  }
    } else {
      //const float incrementFactor = 1.1;
	  //const float newLR = incrementFactor*solver1->getBaseLR();
  	  const float incrementFactor = baseLR/100;
  	  const float newLR = solver1->getBaseLR() + incrementFactor;
      if (newLR <= (baseLR*10)) {
        solver1->setBaseLR(newLR);
        solver2->setBaseLR(newLR);
        CombinedSolver->setBaseLR(newLR);
        LOG(INFO) << "Increase LR";
	    LOG(INFO) << "Current LR is: " << solver1->getBaseLR();
      }
    }
  }
*/
  float Rsquare;
  float alphaOut = QuadraticEstimate(alphas,losses, c0,c1,c2,Rsquare); //minimal alpha by parabolic estimation
  lastKnownLoss = c0 * alphaOut * alphaOut + c1 * alphaOut + c2;
  LOG(INFO) << "outside a = " << c0 << " b = " << c1 << " c = " << c2;
  LOG(INFO) << "outside Last known loss: " << lastKnownLoss;
  LOG(INFO) << "outside alphatOut: " << alphaOut;
  if ((lastKnownLoss < 0) || (alphaOut > maxAlpha) || (alphaOut < minAlpha) || (Rsquare < 0.95)) {
    lastKnownLoss = opt_loss;
    alphaOut = opt_alpha;
  }
  LOG(INFO) << "Exit Last known loss: " << lastKnownLoss;
  return alphaOut;
  alphas.clear();
  losses.clear();
}

float CheckLinearCombination2(shared_ptr<caffe::Solver<float> > solver1,
		shared_ptr<caffe::Solver<float> > solver2,
		shared_ptr<caffe::Solver<float> > CombinedSolver,
		float & lastKnownLoss, const float baseLR, const int iteration) {
  // return minimum alpha

  const float step_size = 0.2;
  const float minAlpha = -1.0;
  float opt_alpha = minAlpha;
  float opt_loss = 0;
  const float maxAlpha = 4;
  const vector<shared_ptr<Blob<float> > >& net_params1 = solver1->net()->params();
  const vector<shared_ptr<Blob<float> > >& net_params2 = solver2->net()->params();
  const vector<shared_ptr<Blob<float> > >& net_params = CombinedSolver->net()->params();
  CHECK(net_params1.size() == net_params2.size()) << "Networks dimensions mismatch";
  vector<float> alphas, losses;
  alphas.clear();
  losses.clear();
  LOG(INFO) << "Starting linear combination test iterations";
  float loss_0;
  float loss_1;
  for (float alpha = minAlpha; alpha <= maxAlpha; alpha += step_size) {
    // Part A - make new combined network
	for (int i = 0; i < net_params1.size(); ++i) {
  	  CHECK(net_params1[i]->count() == net_params2[i]->count()) << "Layer dimensions mismatch";
      int n = net_params1[i]->count();
      // check contraction
      Blob<float> tempVec;
      tempVec.Reshape(n,1,1,1);
      switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe::caffe_copy(n,net_params2[i]->cpu_data(), net_params[i]->mutable_cpu_data());
          caffe::caffe_cpu_axpby(n, alpha, net_params1[i]->cpu_data(), (float)(1.0 - alpha), net_params[i]->mutable_cpu_data());
          break;
        case Caffe::GPU:
          caffe::caffe_copy(n,net_params2[i]->gpu_data(), net_params[i]->mutable_gpu_data());
          caffe::caffe_gpu_axpby(n, alpha, net_params1[i]->gpu_data(), (float)(1.0 - alpha), net_params[i]->mutable_gpu_data());
          break;
      }
    }
	// Part B - testLoss and trainLoss
	//float TestLoss = CombinedSolver->TestLoss(alpha); // to check the algorithm we omit the test loss
	float TrainLoss = CombinedSolver->TrainLoss(alpha);
	if ((alpha > (0.0 - step_size/2)) && (alpha < (0.0 + step_size/2)))
	  loss_0 = TrainLoss;
	else if (((alpha > (1.0 - step_size/2)) && (alpha < (1.0 + step_size/2))))
	  loss_1 = TrainLoss;

	alphas.push_back(alpha);
	losses.push_back(TrainLoss);
	if (alpha == minAlpha) {
	  opt_loss = TrainLoss;
	} else if (TrainLoss < opt_loss) {
	  opt_loss = TrainLoss;
	  opt_alpha = alpha;
	}
/*
	LOG(INFO) << "Iteration #" << solver1->iter() << " for alpha = " << std::fixed <<
			std::setprecision(2)<< alpha <<
			", Test Loss = " << std::fixed << std::setprecision(8) << TestLoss <<
			" and Train Loss = " << std::fixed << std::setprecision(8) << TrainLoss <<
			" and Train 10000 Loss = " << std::fixed << std::setprecision(8) << Solver10000->TrainLoss(alpha,100) <<
			" and Train 5000 Loss = " <<std::fixed << std::setprecision(8) << Solver5000->TrainLoss(alpha,50) <<
			" and Train 1000 Loss = " << std::fixed << std::setprecision(8) << Solver1000->TrainLoss(alpha,10);
*/
	LOG(INFO) << "Iteration #" << solver1->iter() << " for alpha = " << std::fixed <<
			std::setprecision(2)<< alpha <<
			" and Train Loss = " << std::fixed << std::setprecision(8) << TrainLoss;

  }
  // Find best fit quadratic function to data
  float c0,c1,c2;
  for (int i=0; i< alphas.size(); i++) {
	  LOG(INFO) << alphas[i] << " " << losses[i];
  }
	LOG(INFO) << "Last known loss: " << lastKnownLoss;
	LOG(INFO) << "Loss_0: " << loss_0;
	LOG(INFO) << "Loss_1: " << loss_1;

  float Rsquare;
  float alphaOut = QuadraticEstimate(alphas,losses, c0,c1,c2,Rsquare); //minimal alpha by parabolic estimation
  lastKnownLoss = c0 * alphaOut * alphaOut + c1 * alphaOut + c2;
  LOG(INFO) << "outside a = " << c0 << " b = " << c1 << " c = " << c2;
  LOG(INFO) << "outside Last known loss: " << lastKnownLoss;
  LOG(INFO) << "outside alphatOut: " << alphaOut;
  if (Rsquare < 0.8) {
    lastKnownLoss = opt_loss;
    alphaOut = opt_alpha;
  }
  LOG(INFO) << "Exit Last known loss: " << lastKnownLoss;
  return alphaOut;
  alphas.clear();
  losses.clear();
}

void MergeNetworks(shared_ptr<caffe::Solver<float> > solver1,
		shared_ptr<caffe::Solver<float> > solver2, float alpha) {
  const vector<shared_ptr<Blob<float> > >& net_params1 = solver1->net()->params();
  const vector<shared_ptr<Blob<float> > >& net_params2 = solver2->net()->params();
  for (int i = 0; i < net_params1.size(); ++i) {
    CHECK(net_params1[i]->count() == net_params2[i]->count()) << "Layer dimensions mismatch";
    int n = net_params1[i]->count();
    switch (Caffe::mode()) {
      case Caffe::CPU:
	    caffe::caffe_cpu_axpby(n, alpha, net_params1[i]->cpu_data(), (float)(1.0 - alpha), net_params2[i]->mutable_cpu_data());
	    caffe::caffe_copy(n,net_params2[i]->cpu_data(), net_params1[i]->mutable_cpu_data());
	    caffe::caffe_memset(n, .0,solver1->history()[i]->mutable_cpu_data());
	    break;
      case Caffe::GPU:
	    caffe::caffe_gpu_axpby(n, alpha, net_params1[i]->gpu_data(), (float)(1.0 - alpha), net_params2[i]->mutable_gpu_data());
	    caffe::caffe_copy(n,net_params2[i]->gpu_data(), net_params1[i]->mutable_gpu_data());
	    caffe::caffe_gpu_memset(n, .0,solver1->history()[i]->mutable_gpu_data());
	    break;
    }
  }
}

void initBuffer(shared_ptr<caffe::Solver<float> > solver ,vector<shared_ptr<Blob<float> > > & current_state) {
  const vector<shared_ptr<Blob<float> > >& net_params = solver->net()->params();
  current_state.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<float>* net_param = net_params[i].get();
    current_state.push_back(shared_ptr<Blob<float> >(new Blob<float>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}

void SaveWeights(shared_ptr<caffe::Solver<float> > solver ,vector<shared_ptr<Blob<float> > > & current_state) {
  const vector<shared_ptr<Blob<float> > >& net_params = solver->net()->params();
  switch (Caffe::mode()) {
    case Caffe::CPU:
      for (int i = 0; i < net_params.size(); ++i) {
        int n = net_params[i]->count();
        caffe::caffe_copy(n, net_params[i]->cpu_data(), current_state[i]->mutable_cpu_data());
        caffe::caffe_copy(n, solver->history()[i]->cpu_data(), current_state[i]->mutable_cpu_diff());
      }
      break;
    case Caffe::GPU:
      for (int i = 0; i < net_params.size(); ++i) {
        int n = net_params[i]->count();
        caffe::caffe_copy(n, net_params[i]->gpu_data(), current_state[i]->mutable_gpu_data());
        caffe::caffe_copy(n, current_state[i]->gpu_diff(), solver->history()[i]->mutable_gpu_data());
      }
      break;
  }
}

void LoadWeights(shared_ptr<caffe::Solver<float> > solver ,vector<shared_ptr<Blob<float> > > & current_state) {
  const vector<shared_ptr<Blob<float> > >& net_params = solver->net()->params();
  switch (Caffe::mode()) {
	case Caffe::CPU:
	  for (int i = 0; i < net_params.size(); ++i) {
		int n = net_params[i]->count();
		caffe::caffe_copy(n, current_state[i]->cpu_data(), net_params[i]->mutable_cpu_data());
	  }
	  break;
	case Caffe::GPU:
	  for (int i = 0; i < net_params.size(); ++i) {
		int n = net_params[i]->count();
		caffe::caffe_copy(n, current_state[i]->gpu_data(), net_params[i]->mutable_gpu_data());
	  }
	  break;
  }

}



int train2models() {
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  CHECK(!FLAGS_snapshot2.size() || !FLAGS_weights2.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  int start_iteration = FLAGS_start_iteration;
  int iter_between_map = FLAGS_iteration_distance;
  caffe::SolverParameter solver_param1;
  caffe::SolverParameter solver_param2;
  caffe::SolverParameter solver_param_Comb;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param1);
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver2, &solver_param2);
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solverComb, &solver_param_Comb);
  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param1.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param1.device_id();
  }
	  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param_Comb));
  shared_ptr<caffe::Solver<float> >
    solver1(caffe::GetSolver<float>(solver_param1));
  shared_ptr<caffe::Solver<float> >
    solver2(caffe::GetSolver<float>(solver_param2));
  shared_ptr<caffe::Solver<float> >
    CombinedSolver(caffe::GetSolver<float>(solver_param_Comb));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver1->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver1, FLAGS_weights);
    solver1->Solve();
  } else {
	int max_iteration = solver_param1.max_iter();

    solver->Solve(start_iteration);
	std::string snapshotName1(solver->SnapshotReturnName());
	//solver->Solve(0);
	//solver->Solve(1);
    std::string snapshotName2(solver->SnapshotReturnName());
	// first parallel itertion - workers setup
    //vector<shared_ptr<Blob<float> > > current_state;
    //initBuffer(solver1, current_state);
    //SaveWeights(solver1, current_state);
    //LoadWeights(solver2, current_state);
    //solver1->Solve(1);
    //current_state.clear();
    const int epoch_length = 500; // for cifar and current batch size/ for mnist 100
    //const int epoch_length = 600; // for halved db
    //iter_between_map = epoch_length; // for optimization purpose - not normal check
    bool optimizationFlag = false;
    float lastKnownLoss = -1.0;
    const float baseLR = solver1->getBaseLR();
	for (int parallel_iter = 0; parallel_iter < ((max_iteration-start_iteration-1)/iter_between_map); ++parallel_iter) {
      if (parallel_iter == 0) {
        solver1->Solve(iter_between_map, snapshotName1.c_str());
        //solver2->Solve(iter_between_map, snapshotName2.c_str());

/*
      } else if (!optimizationFlag) {
        solver1->Solve(iter_between_map);
        solver2->Solve(iter_between_map);
      } else {
        solver1->Solve(iter_between_map, snapshotName1.c_str());
        solver2->Solve(iter_between_map);
        optimizationFlag = false;
      }

*/	  } else {
        solver1->Solve(iter_between_map);
        //solver2->Solve(iter_between_map);
      }
      //CompareNets(solver1, solver2); // for optimization purpose - not normal check
      //if (parallel_iter % epoch_length == 0) {
        //for optimization purpose - not normal check
      float alpha;
// merge part
      //alpha = CheckLinearCombination(solver1, solver2, CombinedSolver, lastKnownLoss, baseLR, parallel_iter);
      //LOG(INFO) << "optimal alpha: " << alpha;
      //LOG(INFO) << "Last known loss: " << lastKnownLoss;
      //solver1->TestCheck("Solver1Before");
      //solver2->TestCheck("Solver2Before");
      //MergeNetworks(solver1, solver2, alpha);
      //solver1->TestCheck("AfterMerge");
// linear part

      alpha = CheckLinearCombination2(solver1, solver, CombinedSolver, lastKnownLoss, baseLR, parallel_iter);
      MergeNetworks(solver1, solver, alpha);
      LOG(INFO) << "optimal alpha: " << alpha;
      solver1->TestCheck("AfterMerge");


      //std::string snapshotName(solver1->SnapshotReturnName());
      //solver->Solve(1, snapshotName.c_str());
      //snapshotName1 = solver->SnapshotReturnName();
      optimizationFlag = true;
      //}
	}

	int iterations_left = (max_iteration - start_iteration - 1) -
			((max_iteration-start_iteration-1)/iter_between_map) * iter_between_map;
	if (iterations_left > 0) {
	  solver1->Solve(iterations_left);
	  //solver2->Solve(iterations_left);
      //CompareNets(solver1,solver2); // for optimization purpose - not normal check
	}
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  // checks if two workers are in place
  if (FLAGS_solver2.size() > 0)
    return train2models();
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
