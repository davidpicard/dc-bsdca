/**
 * 
 */
package budget;

import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.io.CsvImporter;
import net.jkernelmachines.evaluation.AccuracyEvaluator;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * @author picard
 *
 */
public class DoubleDCBudgetSDCATrainTestCSV {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		List<TrainingSample<double[]>> train = null;
		List<TrainingSample<double[]>> test = null;
		DoubleDCBudgetSDCA cls = new DoubleDCBudgetSDCA();
		int e = 1;

		for (int i = 0; i < args.length; i++) {

			if ("-train".equalsIgnoreCase(args[i])) {
				train = CsvImporter.importFromFile(args[++i], "[ ]+", 0);
				// DataPreProcessing.centerList(train);
				// DataPreProcessing.normalizeList(train);
				Collections.shuffle(train);
				System.out.println("train imported from " + args[i] + ": "
						+ train.size() + " samples of dim "
						+ train.get(0).sample.length);
			} else if ("-test".equalsIgnoreCase(args[i])) {
				test = CsvImporter.importFromFile(args[++i], "[ ]+", 0);
				// DataPreProcessing.centerList(test);
				// DataPreProcessing.normalizeList(test);
				System.out.println("test imported from " + args[i] + ": "
						+ test.size() + " samples of dim "
						+ test.get(0).sample.length);
			} else if ("-C".equals(args[i])) {
				cls.setC(Double.parseDouble(args[++i]));
			} else if ("-v".equalsIgnoreCase(args[i])) {
				DebugPrinter.setDebugLevel(Integer.parseInt(args[++i]));
			} else if ("-c".equals(args[i])) {
				cls.setCapacity(Double.parseDouble(args[++i]));
			} else if ("-b".equalsIgnoreCase(args[i])) {
				cls.setBudget(Integer.parseInt(args[++i]));
			} else if ("-e".equalsIgnoreCase(args[i])) {
				e = Integer.parseInt(args[++i]);
			} else if ("-g".equalsIgnoreCase(args[i])) {
				cls.setKernel(new DoubleGaussL2(Double.parseDouble(args[++i])));
			} else if ("-k".equalsIgnoreCase(args[i])) {
				cls.setK(Integer.parseInt(args[++i]));
			} else if ("-a".equalsIgnoreCase(args[i])) {
				cls.setAlpha(Double.parseDouble(args[++i]));
			}
		}

		if (train == null || test == null) {
			System.out
					.println("usage: DoubleDCBudgetSDCATrainTestCSV -train train.csv -test test.csv [-C <C> -v <verbose> -b <budget> -e <epochs> -g <gamma> -k <clusters> -a <update_kmeans>]");
			return;
		}

		AccuracyEvaluator<double[]> ae = new AccuracyEvaluator<>();
		ae.setClassifier(cls);
		ae.setTrainingSet(train);
		ae.setTestingSet(test);
		ae.evaluate();

		System.out.println("acc: " + ae.getScore());

	}

}
