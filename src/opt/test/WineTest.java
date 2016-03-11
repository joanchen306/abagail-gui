package opt.test;

import dist.test.JGraph;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import func.nn.feedfwd.*;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import util.linalg.Vector;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import javafx.scene.control.TextArea;
/**
 * Created by osama on 3/6/16.
 */
public class WineTest extends Application{

    static BufferedWriter writer;

    private static int counter;

    private static double temp;
    private static double cooling;
    private static int populationSize;
    private static int toMate;
    private static int toMutate;
    private static int restarts;
    private String runInstruc;
    private static ArrayList<double[]> runs;

    static double trainError;
    static double testError;
    private static String[] labels;
  private static Set<String> unique_labels;

  private static Instance[] allInstances;
  private static Map<String, double[]> bitvector_mappings;

//  These fields are hardcoded, depending on your problem

//  Filename for your csv dataset
    // "src/opt/test/wine-train.txt"
  private static String filename;
    private static String resultFile;
//  How many examples your have
    //3429
  private static int num_examples;

//  How many fields are attributes. This is the number of columns you have minus 1.
//  The last column of your CSV will be used as the classification.
    //11
  private static int num_attributes;

//  Number of input nodes is the same as the number of attributes for your problem
  private static int inputLayer = num_attributes;

//  TODO: Manipulate these value. They are your hyper parameters for the Neural Network
  private static int hiddenLayer;
  private static int trainingIterations;

//  This is determined later
  private static int outputLayer;


  private static BackPropagationNetworkFactory bpfactory = new BackPropagationNetworkFactory();
  private static FeedForwardNeuralNetworkFactory fffactory = new FeedForwardNeuralNetworkFactory();
  private static ErrorMeasure measure = new SumOfSquaresError();

  private static DecimalFormat df = new DecimalFormat("0.000");

//  Used by backprop
  private static double backprop_threshold = 1e-10;

//  Train and Test sets
  private static DataSet trainSet;
  private static DataSet testSet;

  private static final double PERCENT_TRAIN = 0.7;

  //  For cross fold validation
  private static final int K = 10;

  private static List<Instance[]> folds;

    private static ArrayList<Double> bp_trainingErr = new ArrayList<>();
    private static ArrayList<Double> bp_testingErr = new ArrayList<>();

    private static ArrayList<Double> rhc_trainingErr = new ArrayList<>();
    private static ArrayList<Double> rhc_testingErr = new ArrayList<>();

    private static ArrayList<Double> sa_trainingErr = new ArrayList<>();
    private static ArrayList<Double> sa_testingErr = new ArrayList<>();

    private static ArrayList<Double> ga_trainingErr = new ArrayList<>();
    private static ArrayList<Double> ga_testingErr = new ArrayList<>();


  public static void main(String[] args) {
      launch(args);
  }

  /**
   * Randomized Hill Climbing with Random restarts
   */
  public static void runRHC() throws IOException {

    System.out.println("===========Randomized Hill Climbing=========");
      writer.write("===========Randomized Hill Climbing=========");
      writer.newLine();

    FeedForwardNetwork[] nets = new FeedForwardNetwork[K];
    NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
    OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];

    double[] validationErrors = new double[nets.length];
    double[] trainErrors = new double[nets.length];

    double starttime = System.nanoTime();;
    double endtime;

    for (int i = 0; i < nets.length; i++) {

      Instance[] validation = getValidationFold(folds, i);
      Instance[] trainFolds = getTrainFolds(folds, i);
      DataSet trnfoldsSet = new DataSet(trainFolds);

      nets[i] = fffactory.createClassificationNetwork(
          new int[] {inputLayer, hiddenLayer, outputLayer});
      nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, nets[i], measure);
      oas[i] = new RandomizedHillClimbing(nnops[i], restarts);

      FeedForwardNetwork ffNet = nets[i];

//      TODO: Vary the number of iterations as needed for your results
      train(oas[i], nets[i], trainingIterations);


      validationErrors[i] = evaluateNetwork(ffNet, validation);
      System.out.printf("Fold: %d\tError: %f%%%n", i+1, validationErrors[i] * 100);
      trainErrors[i] = evaluateNetwork(ffNet, trainFolds);
    }


    int best_index = -1;
    double min = Double.MAX_VALUE;
    for (int j = 0; j < validationErrors.length; j++) {
      if (validationErrors[j] < min) {
        best_index = j;
        min = validationErrors[j];
      }
    }

    FeedForwardNetwork bestNet = nets[best_index];
    double validationError = validationErrors[best_index];
      trainError = trainErrors[best_index];
    testError = evaluateNetwork(bestNet, testSet.getInstances());


    System.out.printf("%nMin Validation Error: %f%% %n", validationError * 100);
    System.out.printf("Training Error: %f%% %n", trainError * 100);
    System.out.printf("Test Error: %f%% %n", testError * 100);


    endtime = System.nanoTime();
    double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
    time_elapsed /= Math.pow(10,9);
    System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));
      writer.write("Min Validation Error: " + Double.toString(validationError*100));
      writer.newLine();
      writer.write("Training Error: " + Double.toString(trainError*100));
      writer.newLine();
      writer.write("Test Error: " + Double.toString(testError*100));
      writer.newLine();
      writer.write("Time Elapsed: " + df.format(time_elapsed));
      writer.newLine();


  }

  /**
   * Run simulated annealing
   */
  public static void runSA() throws IOException {

//    TODO: Tweak these params for SA

    System.out.println("===========Simulated Annealing=========");
      writer.write("===========Simulated Annealing=========");
      writer.newLine();
      writer.write("temperature: " + Double.toString(temp));
      writer.newLine();
      writer.write("cooling: " + Double.toString(cooling));
      writer.newLine();


    FeedForwardNetwork[] nets = new FeedForwardNetwork[K];
    NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
    OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];

    double[] validationErrors = new double[nets.length];
    double[] trainErrors = new double[nets.length];

    double starttime = System.nanoTime();;
    double endtime;

    for (int i = 0; i < nets.length; i++) {

      Instance[] validation = getValidationFold(folds, i);
      Instance[] trainFolds = getTrainFolds(folds, i);
      DataSet trnfoldsSet = new DataSet(trainFolds);

      nets[i] = fffactory.createClassificationNetwork(
          new int[] {inputLayer, hiddenLayer, outputLayer});
      nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, nets[i], measure);
      oas[i] = new SimulatedAnnealing(temp, cooling, nnops[i]);

      FeedForwardNetwork ffNet = nets[i];

//      TODO: Vary the number of iterations as needed for your results
      train(oas[i], nets[i], trainingIterations);

      validationErrors[i] = evaluateNetwork(ffNet, validation);
      System.out.printf("Fold: %d\tError: %f%%%n", i+1, validationErrors[i] * 100);
      trainErrors[i] = evaluateNetwork(ffNet, trainFolds);
    }


    int best_index = -1;
    double min = Double.MAX_VALUE;
    for (int j = 0; j < validationErrors.length; j++) {
      if (validationErrors[j] < min) {
        best_index = j;
        min = validationErrors[j];
      }
    }

    FeedForwardNetwork bestNet = nets[best_index];
    double validationError = validationErrors[best_index];
      trainError = trainErrors[best_index];
      testError = evaluateNetwork(bestNet, testSet.getInstances());


    System.out.printf("%nMin Validation Error: %f%% %n", validationError * 100);
    System.out.printf("Training Error: %f%% %n", trainError * 100);
    System.out.printf("Test Error: %f%% %n", testError * 100);

    endtime = System.nanoTime();
    double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
    time_elapsed /= Math.pow(10,9);
    System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));
      writer.write("Min Validation Error: " + Double.toString(validationError*100));
      writer.newLine();
      writer.write("Training Error: " + Double.toString(trainError*100));
      writer.newLine();
      writer.write("Test Error: " + Double.toString(testError*100));
      writer.newLine();
      writer.write("Time Elapsed: " + df.format(time_elapsed));
      writer.newLine();
  }

  /**
   * Run genetic algorithms.
   */
  public static void runGA() throws IOException {

//    TODO: Tweak these params for GA

    System.out.println("===========Genetic Algorithms=========");
      writer.write("===========Genetic Algorithms=========");
      writer.newLine();
      writer.write("population size: " + Integer.toString(populationSize));
      writer.newLine();
      writer.write("Mate: " + Integer.toString(toMate));
      writer.newLine();
      writer.write("Mutation: " + Integer.toString(toMutate));
      writer.newLine();


      FeedForwardNetwork[] nets = new FeedForwardNetwork[K];
    NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
    OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];

    double[] validationErrors = new double[nets.length];
    double[] trainErrors = new double[nets.length];

    double starttime = System.nanoTime();;
    double endtime;

    for (int i = 0; i < nets.length; i++) {

      Instance[] validation = getValidationFold(folds, i);
      Instance[] trainFolds = getTrainFolds(folds, i);
      DataSet trnfoldsSet = new DataSet(trainFolds);

      nets[i] = fffactory.createClassificationNetwork(
          new int[] {inputLayer, hiddenLayer, outputLayer});
      nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, nets[i], measure);
      oas[i] = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnops[i]);

      FeedForwardNetwork ffNet = nets[i];

//      TODO: Vary the number of iterations as needed for your results
      train(oas[i], nets[i], trainingIterations);

      validationErrors[i] = evaluateNetwork(ffNet, validation);
      System.out.printf("Fold: %d\tError: %f%%%n", i+1, validationErrors[i] * 100);
      trainErrors[i] = evaluateNetwork(ffNet, trainFolds);
    }


    int best_index = -1;
    double min = Double.MAX_VALUE;
    for (int j = 0; j < validationErrors.length; j++) {
      if (validationErrors[j] < min) {
        best_index = j;
        min = validationErrors[j];
      }
    }

    FeedForwardNetwork bestNet = nets[best_index];
    double validationError = validationErrors[best_index];
    trainError = trainErrors[best_index];
    testError = evaluateNetwork(bestNet, testSet.getInstances());


    System.out.printf("%nMin Validation Error: %f%% %n", validationError * 100);
    System.out.printf("Training Error: %f%% %n", trainError * 100);
    System.out.printf("Test Error: %f%% %n", testError * 100);

    endtime = System.nanoTime();
    double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
    time_elapsed /= Math.pow(10,9);
    System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));
      writer.write("Min Validation Error: " + Double.toString(validationError*100));
      writer.newLine();
      writer.write("Training Error: " + Double.toString(trainError*100));
      writer.newLine();
      writer.write("Test Error: " + Double.toString(testError*100));
      writer.newLine();
      writer.write("Time Elapsed: " + df.format(time_elapsed));
      writer.newLine();
  }


  /**
   * This method will run Backpropagation using each
   * combination of (K-1) folds for training, and the Kth fold for validation. Once the model
   * with the lowest validation set error is found, that is used as the "best" model and the
   * training and test errors on that model are recorded.
   */
  public static void runBackprop() throws IOException {

    System.out.println("===========Backprop=========");
      writer.write("=======Back Propagation=====");
      writer.newLine();

    BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
    double[] validationErrors = new double[nets.length];
    double[] trainErrors = new double[nets.length];

    double starttime = System.nanoTime();
    double endtime;

    for (int i = 0; i < nets.length; i++) {
      Instance[] validation = getValidationFold(folds, i);
      Instance[] trainFolds = getTrainFolds(folds, i);
      DataSet trnfoldsSet = new DataSet(trainFolds);

      nets[i] = bpfactory.createClassificationNetwork(
          new int[]{inputLayer, hiddenLayer, outputLayer});
      BackPropagationNetwork backpropNet = nets[i];

      ConvergenceTrainer trainer = new ConvergenceTrainer(
          new BatchBackPropagationTrainer(trnfoldsSet, backpropNet, new SumOfSquaresError(),
              new RPROPUpdateRule()),
          backprop_threshold, trainingIterations);

      trainer.train();

      validationErrors[i] = evaluateNetwork_bb(backpropNet, validation);
      System.out.printf("Fold: %d\tError: %f%%%n", i+1, validationErrors[i] * 100);
      trainErrors[i] = evaluateNetwork_bb(backpropNet, trainFolds);
    }

    int best_index = -1;
    double min = Double.MAX_VALUE;
    for (int j = 0; j < validationErrors.length; j++) {
      if (validationErrors[j] < min) {
        best_index = j;
        min = validationErrors[j];
      }
    }

    BackPropagationNetwork bestNet = nets[best_index];
    double validationError = validationErrors[best_index];
    trainError = trainErrors[best_index];
    testError = evaluateNetwork(bestNet, testSet.getInstances());


    System.out.println("\nConvergence in " + trainingIterations + " iterations");

    System.out.printf("%nMin Validation Error: %f%% %n", validationError * 100);
    System.out.printf("Training Error: %f%% %n", trainError * 100);
    System.out.printf("Test Error: %f%% %n", testError * 100);

    endtime = System.nanoTime();
    double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
    time_elapsed /= Math.pow(10,9);
    System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));
      writer.write("Min Validation Error: " + Double.toString(validationError*100));
      writer.newLine();
      writer.write("Training Error: " + Double.toString(trainError*100));
      writer.newLine();
      writer.write("Test Error: " + Double.toString(testError*100));
      writer.newLine();
      writer.write("Time Elapsed: " + df.format(time_elapsed));
      writer.newLine();

  }

  /**
   * Train a given optimization problem for a given number of iterations. Called by RHC, SA, and
   * GA algorithms
   * @param oa the optimization algorithm
   * @param network the network that corresponds to the randomized optimization problem. The
   *                optimization algorithm will determine the best weights to try using with this
   *                network and assign those weights
   * @param iterations the number of training iterations
   */
  private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, int
      iterations) {

      char[] animationChars = new char[] {'|', '/', '-', '\\'};
    for(int i = 0; i < iterations; i++) {
        System.out.print("Processing: " + counter + " run/"
                + i + " iterations " + animationChars[i % 4] + "\r");
      oa.train();
    }
      Instance optimalWeights = oa.getOptimal();
    network.setWeights(optimalWeights.getData());
  }

  /**
   * Given a network and instances, the output of the network is evaluated and a decimal value
   * for error is given
   * @param network the BackPropagationNetwork with weights already initialized
   * @param data the instances to be evaluated against
   * @return
   */
  public static double evaluateNetwork(FeedForwardNetwork network, Instance[] data) {

    double num_incorrect = 0;
    double error = 0;

    for (int j = 0; j < data.length; j++) {
      network.setInputValues(data[j].getData());
      network.run();

      Vector actual = data[j].getLabel().getData();
      Vector predicted = network.getOutputValues();


      boolean mismatch = ! isEqualOutputs(actual, predicted);

      if (mismatch) {
        num_incorrect += 1;
      }

    }

    error = num_incorrect / data.length;
    return error;

  }

   /**
   * Given a network and instances, the output of the network is evaluated and a decimal value
   * for error is given
   * @param network the BackPropagationNetwork with weights already initialized
   * @param data the instances to be evaluated against
   * @return
   */
  public static double evaluateNetwork_bb(BackPropagationNetwork network, Instance[] data) {

    double num_incorrect = 0;
    double error = 0;

    for (int j = 0; j < data.length; j++) {
      network.setInputValues(data[j].getData());
      network.run();

      Vector actual = data[j].getLabel().getData();
      Vector predicted = network.getOutputValues();


      boolean mismatch = ! isEqualOutputs(actual, predicted);

      if (mismatch) {
        num_incorrect += 1;
      }

    }

    error = num_incorrect / data.length;
    return error;

  }

  /**
   * Compares two bit vectors to see if expected bit vector is most likely to be the same
   * class as the actual bit vector
   * @param actual
   * @param predicted
   * @return
   */
  private static boolean isEqualOutputs(Vector actual, Vector predicted) {

    int max_at = 0;
    double max = 0;

//    Where the actual max should be
    int actual_index = 0;

    for (int i = 0; i < actual.size(); i++) {
      double aVal = actual.get(i);

      if (aVal == 1.0) {
        actual_index = i;
      }

      double bVal = predicted.get(i);

      if (bVal > max) {
        max = bVal;
        max_at = i;
      }
    }

    return actual_index == max_at;

  }

  /**
   * Reads a file formatted as CSV. Takes the labels and adds them to the set of labels (which
   * later helps determine the length of bit vectors). Records real-valued attributes. Turns the
   * attributes and labels into bit-vectors. Initializes a DataSet object with these instances.
   */
  private static void initializeInstances() {

    System.out.println("initialing...");


    double[][] attributes = new double[num_examples][];

    labels = new String[num_examples];
    unique_labels = new HashSet<>();


//    Reading dataset
    try {
      BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
        System.out.println(filename);
        String splitter = ",";

//      You don't need these headers, they're just the column labels
        if(filename.contains(".csv")) {
        	String useless_headers = br.readLine();
            System.out.println(useless_headers);
            splitter = ";";
        }

      for(int i = 0; i < attributes.length; i++) {
        Scanner scan = new Scanner(br.readLine());
          scan.useDelimiter(splitter);

        attributes[i] = new double[num_attributes];

        for(int j = 0; j < num_attributes; j++) {
          attributes[i][j] = Double.parseDouble(scan.next());
        }

//        This last element is actually your classification, which is assumed to be a string
        labels[i] = scan.next();
        unique_labels.add(labels[i]);
      }
    }
    catch(Exception e) {
      e.printStackTrace();
    }

    System.out.println(unique_labels);


//    Creating a mapping of bitvectors. So "some classification" => [0, 1, 0, 0]
    int distinct_labels = unique_labels.size();
    outputLayer = distinct_labels;

    bitvector_mappings = new HashMap<>();

    int index = 0;
    for (String label : unique_labels) {
      double[] bitvect = new double[distinct_labels];

//      At index, set to 1 for a given string
      bitvect[index] = 1.0;
//      Increment which index will have a bit flipped in next classification
      index++;

      bitvector_mappings.put(label, bitvect);
    }

//    Replaces the label for each instance with the corresponding bit vector for that label
//    This works even for binary classification
    allInstances = new Instance[num_examples];
    for (int i = 0; i < attributes.length; i++) {
      double[] X = attributes[i];
      String label = labels[i];
      double[] bitvect = bitvector_mappings.get(label);
      Instance instance = new Instance(X);
      instance.setLabel(new Instance(bitvect));
      allInstances[i] = instance;
    }
  }

  /**
   * Print out the actual vs expected bit-vector. Used for debugging purposes only
   * @param actual what the example's actual bit vector looks like
   * @param expected what a network output as a bit vector
   */
  public static void printVectors(Vector actual, Vector expected) {
    System.out.print("Actual: [");
    for (int i = 0; i < actual.size(); i++) {
      System.out.printf(" %f", actual.get(i));
    }
    System.out.print(" ] \t Expected: [");

    for (int i = 0; i < expected.size(); i++) {
      System.out.printf(" %f", expected.get(i));
    }
    System.out.println(" ]");
  }

  /**
   * Takes all instances, and randomly orders them. Then, the first PERCENT_TRAIN percentage of
   * instances form the trainSet DataSet, and the remaining (1 - PERCENT_TRAIN) percentage of
   * instances form the testSet DataSet.
   */
  public static void makeTestTrainSets() {

    List<Instance> instances = new ArrayList<>();

    for (Instance instance: allInstances) {
      instances.add(instance);
    }
    Collections.shuffle(instances);

    int cutoff = (int) (instances.size() * PERCENT_TRAIN);

    List<Instance> trainInstances = instances.subList(0, cutoff);
    List<Instance> testInstances = instances.subList(cutoff, instances.size());

    Instance[] arr_trn = new Instance[trainInstances.size()];
    trainSet = new DataSet(trainInstances.toArray(arr_trn));

    Instance[] arr_tst = new Instance[testInstances.size()];
    testSet = new DataSet(trainInstances.toArray(arr_tst));

  }

  /**
   * Given a DataSet of training data, separate the instances into K nearly-equal-sized
   * partitions called folds for K-folds cross validation
   * @param training, the training DataSet
   * @return a list of folds, where each fold is an Instance[]
   */
  public static List<Instance[]> kfolds(DataSet training) {

    Instance[] trainInstances = training.getInstances();

    List<Instance> instances = new ArrayList<>();
    for (Instance instance: trainInstances) {
      instances.add(instance);
    }

    List<Instance[]> folds = new ArrayList<>();

//    Number of values per fold
    int per_fold = (int) Math.floor((double)(instances.size()) / K);

    int start = 0;
    int end = per_fold;

    while (start < instances.size()) {


      List<Instance> foldList = null;

      if (end > instances.size()) {
        end = instances.size();
      }
      foldList = instances.subList(start, end);

      Instance[] fold = new Instance[foldList.size()];
      fold = foldList.toArray(fold);

      folds.add(fold);

      start = end + 1;
      end = start + per_fold;

    }
    return folds;
  }

  /**
   * Given a list of Instance[], this helper combines each arrays contents into one, single
   * output array
   * @param instanceList the list of Instance[]
   * @return the combined array consisting of the contents of each Instance[] in instanceList
   */
  public static Instance[] combineInstances(List<Instance[]> instanceList) {
    List<Instance> combined = new ArrayList<>();

    for (Instance[] fold: instanceList) {

      for (Instance instance : fold) {
        combined.add(instance);
      }
    }

    Instance[] combinedArr = new Instance[combined.size()];
    combinedArr = combined.toArray(combinedArr);
    return combinedArr;
  }

  /**
   * Given a list of folds and an index, it will provide an Instance[] with the combined
   * instances from every fold except for the fold at the given index
   * @param folds the K-folds, a list of Instance[] used as folds for cross-validation
   * @param foldIndex the index of the fold to exclude. That fold is used as the validation set
   * @return the training folds combined into once Instance[]
   */
  public static Instance[] getTrainFolds(List<Instance[]> folds, int foldIndex) {
    List<Instance[]> trainFolds = new ArrayList<>(folds);
    trainFolds.remove(foldIndex);

    Instance[] trnfolds = combineInstances(trainFolds);
    return trnfolds;
  }

  /**
   * Given a list of folds and an index, it will provide an Instance[] to serve as a validation
   * set.
   * @param folds the K-folds, a list of Instance[] used as folds for cross-validation
   * @param foldIndex the index of the fold to use as the validation set
   * @return the validation set
   */
  public static Instance[] getValidationFold(List<Instance[]> folds, int foldIndex) {
    return folds.get(foldIndex);
  }

    @Override
    public void start(Stage primaryStage) throws Exception {

        primaryStage.setTitle("Randomized Optimization with Neural Nets");
        Scene scene = new Scene(new Group(), 400, 600);
        Scene scene2 = new Scene(new Group(), 400, 600);

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Data File");
        final File file = fileChooser.showOpenDialog(primaryStage);


        Label exampleLabel = new Label("Number of Examples: ");
        final TextField exampleInput = new TextField();
        Label attrLabel = new Label("Number of Attributes: ");
        final TextField attrInput = new TextField();
        Label resultLabel = new Label("Save as: ");
        Button saveButton = new Button("...");
        final TextField resultInput = new TextField();
        Button nextButton1 = new Button("NEXT");

        Label hlLabel = new Label("Number of Hidden Layers: ");
        final TextField hlInput = new TextField();
        hlInput.setText("10");
        Label iterLabel = new Label("Number of Iterations: ");
        final TextField iterInput = new TextField();
        iterInput.setText("1000");

        final Pane spring1 = new Pane();
        spring1.minHeightProperty().bind(iterLabel.heightProperty());

        Label rhcLabel = new Label("Randomized Hill Climbing");
        Label reLabel = new Label("Number of Restarts: ");
        final TextField reInput = new TextField();
        reInput.setText("0");

        final Pane spring2 = new Pane();
        spring2.minHeightProperty().bind(iterLabel.heightProperty());

        Label saLabel = new Label("Simulated Annealing");
        Label tempLabel = new Label("Temperature: ");
        final TextField tempInput = new TextField();
        tempInput.setText("1E11");
        Label coolLabel = new Label("Cooling: ");
        final TextField coolInput = new TextField();
        coolInput.setText("0.999");

        final Pane spring3 = new Pane();
        spring3.minHeightProperty().bind(iterLabel.heightProperty());

        Label gaLabel = new Label("Genetic Algorithm");
        Label psLabel = new Label("Population Size: ");
        final TextField psInput = new TextField();
        psInput.setText("200");
        Label mateLabel = new Label("To Mate: ");
        final TextField mateInput = new TextField();
        mateInput.setText("100");
        Label mutLabel = new Label("To Mutate: ");
        final TextField mutInput = new TextField();
        mutInput.setText("20");

        final TextArea record = new TextArea();
        record.setMinSize(350,200);


        Button addButt = new Button("add...");
        Button nextButton2 = new Button("RUN!");

        runs = new ArrayList<double[]>();



        nextButton1.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                primaryStage.setScene(scene2);

                resultFile = resultInput.getText();
                filename = file.getAbsolutePath();
                num_examples = Integer.parseInt(exampleInput.getText());
                num_attributes = Integer.parseInt(attrInput.getText());
                inputLayer = num_attributes;
                exampleInput.setText("");
                attrInput.setText("");


            }
        });

        saveButton.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                DirectoryChooser fileChooser2 = new DirectoryChooser();
                fileChooser2.setTitle("Save To...");
                final File file2 = fileChooser2.showDialog(primaryStage);
                resultFile = file2.getAbsolutePath();
                resultInput.setText(resultFile + "/" + resultInput.getText());


            }
        });

        addButt.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                restarts = Integer.parseInt(reInput.getText());
                hiddenLayer = Integer.parseInt(hlInput.getText());
                trainingIterations = Integer.parseInt(iterInput.getText());
                temp = Double.parseDouble(tempInput.getText());
                cooling = Double.parseDouble(coolInput.getText());
                populationSize = Integer.parseInt(psInput.getText());
                toMate = Integer.parseInt(mateInput.getText());
                toMutate = Integer.parseInt(mutInput.getText());
                runInstruc = "--RS: " + restarts+ " HL " + hiddenLayer + " Iter " + trainingIterations
                        + " Temp " + temp + " Cl " + cooling + " PS " + populationSize
                        + " Ma " + toMate + " Mut " + toMutate;
                record.appendText(runInstruc + "\n");
                runs.add(new double[]{(double) hiddenLayer, (double) trainingIterations, temp, cooling,
                        (double) populationSize, (double) toMate, (double) toMutate, (double) restarts});
                System.out.print(runs);

            }
        });

        nextButton2.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                try {
                    counter = 1;
                    writer = new BufferedWriter(new FileWriter(resultFile));
                    for (double[] r : runs) {
                        restarts = (int) r[7];
                        hiddenLayer = (int) r[0];
                        trainingIterations = (int) r[1];
                        temp = r[2];
                        cooling = r[3];
                        populationSize = (int) r[4];
                        toMate = (int) r[5];
                        toMutate = (int) r[6];
                        run();
                        counter++;
                    }
                    String[] st = new String[2];
                    st[0] = "Training Error";
                    st[1] = "Testing Error";
                    JGraph jg_rhc = new JGraph(st, "Random Restarts");
                    for(int i = 0; i < rhc_trainingErr.size(); i++) {
                        jg_rhc.addToSeries("Training Error", runs.get(i)[7], rhc_trainingErr.get(i));
                        jg_rhc.addToSeries("Testing Error", runs.get(i)[7], rhc_testingErr.get(i));
                    }
                    jg_rhc.createChart();

                    JGraph jg_sa = new JGraph(st, "Temperature");
                    for(int i = 0; i < sa_trainingErr.size(); i++) {
                        jg_rhc.addToSeries("Training Error", runs.get(i)[2], sa_trainingErr.get(i));
                        jg_rhc.addToSeries("Testing Error", runs.get(i)[2], sa_testingErr.get(i));
                    }
                    jg_sa.createChart();

                    JGraph jg_ga = new JGraph(st, "Population Size");
                    for(int i = 0; i < ga_trainingErr.size(); i++) {
                        jg_rhc.addToSeries("Training Error", runs.get(i)[4], ga_trainingErr.get(i));
                        jg_rhc.addToSeries("Testing Error", runs.get(i)[4], ga_testingErr.get(i));
                    }
                    jg_ga.createChart();
                    
                    writer.close();
                    primaryStage.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        GridPane grid1 = new GridPane();
        grid1.setVgap(4);
        grid1.setHgap(10);
        grid1.setPadding(new Insets(5, 5, 5, 5));
        grid1.add(exampleLabel, 0, 0);
        grid1.add(exampleInput, 1, 0);
        grid1.add(attrLabel, 0, 1);
        grid1.add(attrInput, 1, 1);
        grid1.add(resultLabel, 0, 2);
        grid1.add(saveButton, 2, 2);
        grid1.add(resultInput, 1, 2);
        grid1.add(nextButton1, 0, 4);

        GridPane grid2 = new GridPane();
        grid2.setVgap(4);
        grid2.setHgap(10);
        grid2.setPadding(new Insets(5, 5, 5, 5));
        grid2.add(hlLabel, 0, 0);
        grid2.add(hlInput, 1, 0);
        grid2.add(iterLabel, 0, 1);
        grid2.add(iterInput, 1, 1);

        grid2.add(spring1, 0, 2);
        grid2.add(rhcLabel, 0, 3);
        grid2.add(reLabel, 0, 4);
        grid2.add(reInput, 1, 4);

        grid2.add(saLabel, 0, 5);
        grid2.add(tempLabel, 0, 6);
        grid2.add(tempInput, 1, 6);
        grid2.add(coolLabel, 0, 7);
        grid2.add(coolInput, 1, 7);

        grid2.add(spring2, 0, 8);
        grid2.add(gaLabel, 0, 9);
        grid2.add(psLabel, 0, 10);
        grid2.add(psInput, 1, 10);
        grid2.add(mateLabel, 0, 11);
        grid2.add(mateInput, 1, 11);
        grid2.add(mutLabel, 0, 12);
        grid2.add(mutInput, 1, 12);
        grid2.add(record, 0, 13, 3, 1);
        grid2.add(addButt, 0, 14);
        grid2.add(nextButton2, 1, 14);

        Group root1 = (Group) scene.getRoot();
        root1.getChildren().add(grid1);

        Group root2 = (Group) scene2.getRoot();
        root2.getChildren().add(grid2);

        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void run() throws IOException {
        initializeInstances();

        //    Handles cross-fold validation using K folds
        makeTestTrainSets();
        folds = kfolds(trainSet);
        runInstruc = "--HL " + hiddenLayer + " Iter " + trainingIterations
                + " Temp " + temp + " Cl " + cooling + " PS " + populationSize
                + " Ma " + toMate + " Mut " + toMutate;
        writer.write(runInstruc);
        System.out.println(runInstruc);
        writer.newLine();
        writer.write("Iterations: " + trainingIterations);
        writer.newLine();

        runBackprop();
        bp_trainingErr.add(trainError);
        bp_testingErr.add(testError);
        runRHC();
        rhc_trainingErr.add(trainError);
        rhc_testingErr.add(testError);
        runSA();
        sa_trainingErr.add(trainError);
        sa_testingErr.add(testError);
        runGA();
        ga_trainingErr.add(trainError);
        ga_testingErr.add(testError);
    }
}
