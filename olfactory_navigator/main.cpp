#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const double TransientDuration = 500;
// const double RunDuration = 200;
const double StepSize = 0.01;
const double TargetFreq = .1;
const double DistThreshold = .05;
const double DurThreshold = 1000;

// EA params
const int num_evol_runs = 50;
const int POPSIZE = 50;
const int GENS = 50;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Parameter variability modality only
const int Repetitions = 10; 

// Nervous system params
const int N = 3;
const double WR = 16.0; 
const double BR = 16.0; 
const double TMIN = 1; 
const double TMAX = 1; 

int	VectSize = N*N + 2*N;

// HP params
const double Btau = 20;
const double Wtau = 40;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				phen(k) = MapSearchParameter(gen(k), -WR, WR);
				k++;
			}
	}
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunction(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	TVector<double> pastNeuronOutput(1,N);
	TVector<double> CumRateChange(1,N);
	
	// Create the agent
	CTRNN Agent(N,1,.25,Btau,Wtau,WR,BR);

	// Instantiate the nervous system
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Agent.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
	}

	/// Randomize Outputs
    Agent.RandomizeCircuitOutput(0.1, 0.9);

	// Prior way of fitness calculation: count peaks, divide by time
	// Was subject to innacuracy with double-periodicity & discreteness
  	// TVector<double> numberpeaks(1,N);
    // numberpeaks.FillContents(0);
    // TVector<double> freq(1,N);

    // //pass transient
    // for (double time = StepSize; time <= TransientDuration; time += StepSize) {
    //   Agent.EulerStep(StepSize);
    // }

    // TVector<double> onebackNeuronOutput(1,N);
    // for (int i=1;i<=N;i++){
    //   onebackNeuronOutput[i] = Agent.NeuronOutput(i);
    // }
    // TVector<double> twobackNeuronOutput(1,N);
    // twobackNeuronOutput = onebackNeuronOutput;

    // //test for oscillation frequency of each neuron (count the number of peaks)
    // for (double time = StepSize; time<=RunDuration; time += StepSize){
	//   twobackNeuronOutput = onebackNeuronOutput;
    //   for (int i=1;i<=N;i++){
    //     onebackNeuronOutput[i] = Agent.NeuronOutput(i);
    //   }
    //   Agent.EulerStep(StepSize);
    //   for (int i=1;i<=N;i++){
    //     if (Agent.NeuronOutput(i)<onebackNeuronOutput[i] && onebackNeuronOutput[i]>twobackNeuronOutput[i]){
    //       numberpeaks[i] ++;
    //     }
    //   }
    // }
    // // Divide by test duration in seconds
    // for (int i=1;i<=N;i++){
    //   freq[i] = numberpeaks[i]/RunDuration;
	//   perf += 1 - ((1/(TargetFreq*TargetFreq))*(freq[i]-TargetFreq)*(freq[i]-TargetFreq));
	// //   perf += .5;
    // }

	// perf = perf/3; //Average performance across all three neurons

	//New way of calculating fitness: time how long it takes for circuit to leave & return to same(ish) point in state space
	// Pass transient
	for (double time = StepSize; time <= TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
	}

	// Run the circuit to calculate whether the neurons are oscillating or not (according to Beer 2006)(HP may be on or off)
	CumRateChange.FillContents(0.0);
	for (double time = StepSize; time <= 50; time += StepSize) {
		for (int i = 1; i <= N; i += 1) {
			pastNeuronOutput[i] = Agent.NeuronOutput(i);
		}
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i]));
		}
	}
	int OscillationFlag = 0;
	for (int i = 1; i <= N; i += 1) {
		if (CumRateChange[i]> 0.05)
		{
			OscillationFlag = 1;
		}
	}

	// Only continue fitness if at least one neuron oscillating
	if (OscillationFlag == 1)
	{
		//cout << "B = " << PlasticBoundary << " all oscillating" << endl;
		// Run the circuit to calculate the frequency of oscillation
		// 1. Record the current N-dimensional state
		TVector<double> goalState(1,N);
		for (int i = 1; i <= N; i += 1) 
		{
			// goalState[i] = Agent.NeuronOutput(i);
			goalState[i] = Agent.NeuronState(i);
		}
		// 2. Integrate the system until it's far enough from the starting state
		double dist = 0.0;
		double time = 0.0;
		while  ((dist < DistThreshold) && (time < DurThreshold))
		{
			Agent.EulerStep(StepSize);
			// Re-calculate Euclidean distance with state
			dist = 0;
			for (int i = 1; i <= N; i += 1) 
			{
				// dist += pow(goalState(i) - Agent.NeuronOutput(i), 2);
				dist += pow(goalState(i) - Agent.NeuronState(i), 2);
			}
			dist = sqrt(dist);
			// Update time
			time += StepSize;
		}	

		// If it left in a decent time (meaning it was truly oscillating), then keep going
		if (time < DurThreshold)
		{
			// 3. Integrate the system until it's close enough again! (or until a reasonable length of time runs out)
			time = 0.0;
			while ((dist >= DistThreshold) && (time < DurThreshold))
			{
				Agent.EulerStep(StepSize);
				// Re-calculate Euclidean distance with state
				dist = 0;
				for (int i = 1; i <= N; i += 1) 
				{
					// dist += pow(goalState(i) - Agent.NeuronOutput(i), 2);
					dist += pow(goalState(i) - Agent.NeuronState(i), 2);					
				}
				dist = sqrt(dist);
				// Update time
				time += StepSize;
			}	
			if (time < DurThreshold)
			{
				double measuredFrequency = 1/time;
				//cout << "B = " << PlasticBoundary << " freq measured = " << measuredFrequency << endl;
				return 1.0 - ((1.0/pow(TargetFreq,2.0))*pow((measuredFrequency-TargetFreq),2.0));
			}
			else
			{
				//cout << "Left initial state but didn't find full cycle" << endl;
				return 0.0;
			}
		}
		else
		{
			//cout << "oscillation detected but did not leave initial state in reasonable time" << endl;
			return 0.0;
		}		
	}
	else
	{
		//cout << "none oscillating" << endl;
		return 0.0;
	}
}

// ------------------------------------
// Behavior
// ------------------------------------
void Behavior(TVector<double> &genotype,ofstream &nfile, ofstream &biasfile, ofstream &weightfile)
{
	RandomState rs;

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agent
	CTRNN Agent(N,1,.25,Btau,Wtau,WR,BR);

	// Instantiate the nervous system

	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	
	// Initialize the state at an output of 0.5 for all neurons in the circuit
	Agent.RandomizeCircuitOutput(0.5, 0.5);

	// Run the circuit with HP on
	for (double time = StepSize; time < TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			nfile << Agent.NeuronOutput(i) << " ";
		}
		nfile << endl;
		for (int i = 1; i <= N; i += 1) {
			biasfile << Agent.NeuronBias(i) << " ";
			for (int j = 1; j <= N; j += 1) {
				weightfile << Agent.ConnectionWeight(i,j) << " ";
			}
		}
		biasfile << endl;
		weightfile << endl;
	}

	// Run the circuit with HP off
	for (int i=1;i<=3;i++){Agent.SetPlasticityBoundary(i,0);}
	for (double time = StepSize; time < TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			nfile << Agent.NeuronOutput(i) << " ";
		}
		nfile << endl;
		for (int i = 1; i <= N; i += 1) {
			biasfile << Agent.NeuronBias(i) << " ";
			for (int j = 1; j <= N; j += 1) {
				weightfile << Agent.ConnectionWeight(i,j) << " ";
			}
		}
		biasfile << endl;
		weightfile << endl;
	}

	// DO NOT Re-Instantiate the nervous system (only HP has changed)

	// k = 1;
	// // Time-constants
	// for (int i = 1; i <= N; i++) {
	// 	Agent.SetNeuronTimeConstant(i,phenotype(k));
	// 	k++;
	// }
	// // Bias
	// for (int i = 1; i <= N; i++) {
	// 	Agent.SetNeuronBias(i,phenotype(k));
	// 	k++;
	// }
	// // Weights
	// for (int i = 1; i <= N; i++) {
	// 	for (int j = 1; j <= N; j++) {
	// 		Agent.SetConnectionWeight(i,j,phenotype(k));
	// 		k++;
	// 	}
	// }
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	// Get best individual
	TVector<double> bestVector;
	bestVector = s.BestIndividual();

	// Save the genotype of the best individual to file
	ofstream BestIndividualFile;
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(bestVector, phenotype);

	// Create the agent
	CTRNN Agent(N,1,.25,Btau,Wtau,WR,BR);

	// Instantiate the nervous system

	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	
	// Save the phenotype of the best individual to file
	ofstream BestNSFile;
	BestNSFile.open("best.ns.dat");
	BestNSFile << Agent << endl;
	BestNSFile.close();

	//cout << s.BestPerformance()<< s.;
}

double EvolRun(long IDUM, ofstream &BestFitnessFile){
	// Configure the search

	TSearch s(VectSize);

	s.SetRandomSeed(IDUM);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE);
	s.SetMaxGenerations(GENS);
	s.SetCrossoverProbability(CROSSPROB);
	s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
	s.SetMaxExpectedOffspring(EXPECTED);
	s.SetElitistFraction(ELITISM);
	s.SetSearchConstraint(1);
	s.SetReEvaluationFlag(0); 

	/* Stage 1 */
	// s.SetSearchTerminationFunction(TerminationFunction);
	// s.SetEvaluationFunction(FitnessFunction1); 
	// s.ExecuteSearch();
	/* Stage 2 */
	s.SetSearchTerminationFunction(NULL);
	s.SetEvaluationFunction(FitnessFunction);
	s.ExecuteSearch();

	BestFitnessFile << s.BestPerformance() << endl;

	return s.BestPerformance();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{
	ofstream nfile("neuralactivity.dat");
	ofstream bfile("biastrack.dat");
	ofstream wfile("weighttrack.dat");
	ofstream BestFitnessFile("BestFitness.dat");

	for (int i = 1; i<=num_evol_runs ; i++){
		long IDUM=-time(0);
		EvolRun(IDUM,BestFitnessFile);
		ifstream genefile("best.gen.dat");
		TVector<double> genotype(1, VectSize);
		genefile >> genotype;
		Behavior(genotype,nfile,bfile,wfile);
	}
	nfile.close();
	BestFitnessFile.close();

	// #ifdef PRINTOFILE
	// ofstream file;
	// file.open("evol.dat");
	// cout.rdbuf(file.rdbuf());
	// #endif

	

  return 0;
}
