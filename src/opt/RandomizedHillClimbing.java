package opt;

import shared.Instance;

/**
 * A randomized hill climbing algorithm
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class RandomizedHillClimbing extends OptimizationAlgorithm {
    
    /**
     * The current optimization data
     */
    private Instance cur;
    
    /**
     * The current value of the data
     */
    private double curVal;

    /**
     * number of random restarts
     */
    private int n;

    /**
     * threshold for restarting
     */
    private int thres;
    /**
     * best data
     */
    private Instance best;
    
    /**
     * Make a new randomized hill climbing
     */
    public RandomizedHillClimbing(HillClimbingProblem hcp) {
        super(hcp);
        cur = hcp.random();
        curVal = hcp.value(cur);
        best = cur;
        this.n = 0;
        thres = 0;
    }

    public RandomizedHillClimbing(HillClimbingProblem hcp, int n) {
        super(hcp);
        cur = hcp.random();
        curVal = hcp.value(cur);
        best = cur;
        this.n = n;
        thres = 0;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();
        Instance neigh = hcp.neighbor(cur);
        double neighVal = hcp.value(neigh);
        if (neighVal > curVal) {
            curVal = neighVal;
            cur = neigh;
        } else {
            thres++;
            if(thres >= 15 && n > 0) {
                cur = hcp.random();
                curVal = hcp.value(cur);
                n--;
                thres = 0;
            }
        }

        if (hcp.value(best) < hcp.value(cur)) {
            best = cur;
        }

        return hcp.value(best);
    }

    /**
     * @see opt.OptimizationAlgorithm# getOptimalData()
     */
    public Instance getOptimal() {
        return cur;
    }

}
