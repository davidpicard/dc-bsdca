/**
 * 
 */
package budget;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.classifier.OnlineClassifier;
import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.ListSampleStream;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.type.TrainingSampleStream;
import net.jkernelmachines.util.DebugPrinter;

/**
 * @author picard
 *
 */
public class DoubleDCBudgetSDCA implements OnlineClassifier<double[]> {
	
	int k = 16;
	int m = 64;
	double cap = 1.05;
	double C = 100;
	double alpha = 0.001;
	
	OnlineDoubleKMeans km;
	List<BudgetSDCA<double[]>> cls;
	Kernel<double[]> kernel;
	
	DebugPrinter debug = new DebugPrinter();
	
	public DoubleDCBudgetSDCA() {
		km = new OnlineDoubleKMeans(k);
		km.setAlpha(alpha);
		cls = new ArrayList<BudgetSDCA<double[]>>(k);
		kernel = new DoubleGaussL2(1.0);
	}
	

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {
		TrainingSampleStream<double[]> stream = new ListSampleStream<>(l);
		onlineTrain(stream);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.OnlineClassifier#train(fr.lip6.jkernelmachines.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<double[]> t) {
		km.train(t.sample);
		int ind = (int) km.valueOf(t.sample);
		stat[ind]++;
		while(cls.size() < ind+1) {
			BudgetSDCA<double[]> b = new BudgetSDCA<>(kernel);
			b.setC(C);
			b.setCapacity(cap);
			b.setBudget(m);
			cls.add(b);
		}
		cls.get(ind).train(t);
	}
	
	int stat[];

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.OnlineClassifier#onlineTrain(fr.lip6.jkernelmachines.type.TrainingSampleStream)
	 */
	@Override
	public void onlineTrain(TrainingSampleStream<double[]> stream) {
		km = new OnlineDoubleKMeans(k);
		km.setAlpha(alpha);
		stat = new int[k];
		cls = new ArrayList<>(k);
		TrainingSample<double[]> t;
		int i = 0;
		while ((t = stream.nextSample()) != null) {
			train(t);
			debug.print(2, "\r "+(++i));
		}
		for(BudgetSDCA<double[]> svm : cls) {
			if(!svm.prune()) {
				svm.reprocess();
			}
		}
		debug.println(2, "stat: "+Arrays.toString(stat));
		debug.print(2, "\r                      \r");
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		int id = (int) km.valueOf(e);
		return cls.get(id).valueOf(e);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<double[]> copy() throws CloneNotSupportedException {
		return (DoubleDCBudgetSDCA) this.clone();
	}


	/**
	 * @param parseDouble
	 */
	public void setC(double c) {
		for(BudgetSDCA<double[]> svm : cls) {
			svm.setC(c);
		}
	}


	/**
	 * @param parseDouble
	 */
	public void setCapacity(double c) {
		cap = c;
	}


	/**
	 * @param parseInt
	 */
	public void setBudget(int b) {
		m = b;
	}


	/**
	 * @param doubleGaussL2
	 */
	public void setKernel(Kernel<double[]> k) {
		kernel = k;
	}


	/**
	 * @param parseInt
	 */
	public void setK(int k) {
		this.k = k;
		km = new OnlineDoubleKMeans(k);
	}


	public double getAlpha() {
		return alpha;
	}


	public void setAlpha(double alpha) {
		this.alpha = alpha;
		km.setAlpha(alpha);
	}


}
