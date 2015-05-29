

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Saquib
 */
public class Perceptron {

    boolean stopListEnable;
    List<String> stopWords;
    List<Double> weights;
    List<String> vocabularyList;
    List<List<Integer>> wordCountHamList;
    List<List<Integer>> wordCountSpamList;

    public static String SPAMFOLDERNAME;
    public static String HAMFOLDERNAME;
    public static String SPAMTEST;
    public static String HAMTEST;
    public static String STOPWORDS;

    public Perceptron(String[] args, boolean stopListEnable) throws FileNotFoundException {
        weights = new ArrayList<>();
        wordCountHamList = new ArrayList<>();
        wordCountSpamList = new ArrayList<>();
        vocabularyList = new ArrayList<>();
        vocabularyList.add("");
        

        SPAMFOLDERNAME = args[0] +"/train/spam";
        HAMFOLDERNAME = args[0] +"/train/ham";
        SPAMTEST = args[0]+ "/test/spam";
        HAMTEST = args[0] + "/test/ham";
        STOPWORDS = args[1];

        this.stopListEnable = stopListEnable;

        if (this.stopListEnable) {
            //Populate the stop words
            File stopWordsFile = new File(STOPWORDS);
            stopWords = readStopFile(stopWordsFile);
        } else {
            stopWords = null;
        }
    }
    
     public void populateVocabularyList() throws FileNotFoundException
    {
        vocabularyList = new ArrayList<>();
        File folder = new File(SPAMFOLDERNAME);
        File[] listOfFiles = folder.listFiles();
        List<String> words;
        int i;
        for (File spamFile : listOfFiles) {
            //READ FROM SPAM FILE
            words = readFile(spamFile);
            //COMPUTE THE COUNTS FOR NAIVE BAYES CALCULATION
            for (String word : words) {
                if (!vocabularyList.contains(word)) {
                   vocabularyList.add(word);
                   weights.add(Math.random());
                }
            }
        }
        
        folder = new File(HAMFOLDERNAME);
        listOfFiles = folder.listFiles();
        for (File hamFile : listOfFiles) {
            //READ FROM SPAM FILE
            words = readFile(hamFile);
            //COMPUTE THE COUNTS FOR NAIVE BAYES CALCULATION
            for (String word : words) {
                if (!vocabularyList.contains(word)) {
                   vocabularyList.add(word);
                   weights.add(Math.random());
                }
            }
        }
        
        
    }
    
    
    public void populateWordCount() throws FileNotFoundException
    {   
        File folder = new File(HAMFOLDERNAME);
        File[] listOfFiles = folder.listFiles();
        

        //Populate the wordCount for all the files
        for (File eachFile : listOfFiles) 
        {
            //Read the file and COMPUTE THE X vector for Perceptron weight displacement calculation
            wordCountHamList.add(populateCountFromFile(eachFile));
        }
        
        folder = new File(SPAMFOLDERNAME);
        listOfFiles = folder.listFiles();

        //Populate the wordCount for all the files
        for (File eachFile : listOfFiles) 
        {
            //Read the file and COMPUTE THE X vector for Perceptron weight displacement calculation
            wordCountSpamList.add(populateCountFromFile(eachFile));
        }
    }
    
   
    public List<Double> trainPerceptronClassifier(int noOfIterations, double eenta, List<Double> tempWeights) throws FileNotFoundException {
        
        //Populate the vocabulary list
        populateVocabularyList();
        System.out.println("The size of vocabulary is: " + vocabularyList.size());
        //Populate the wordCount for Spam and Ham files
        populateWordCount();

        if(tempWeights == null)
        {
            tempWeights = new ArrayList<>();
            for(int i=0; i<weights.size(); i++)
            {
                tempWeights.add(weights.get(i)); 
            }
        }
        else
        {
            weights = new ArrayList<>(tempWeights);
        }

        int iterationsSoFar;
        for(double ecount = 0.3 ; ecount < eenta; ecount = ecount+0.03)
        {
           iterationsSoFar = 5;
           trainFromExamples(iterationsSoFar, ecount);
           for(int count = 1; iterationsSoFar <= noOfIterations; count=count+1)
           {
               iterationsSoFar = iterationsSoFar + 1;
               //Train the weights on the ham record
                trainFromExamples(1, ecount);
                double[] value = returnAccuracy();
                System.out.println("Iterations:"+ iterationsSoFar +" Eenta:" + ecount + " Stop Words removed:" + stopListEnable);
                System.out.println("SPAM ACCURACY:" + value[0]);
                System.out.println("HAM ACCURACY:" + value[1]);
           }
           weights = new ArrayList<>(tempWeights);
        }
        return tempWeights;
    }

   
    /**
     * Updates the weight vector for the number of iterations specified
     * 
     * @param noOfIteration
     * @param eenta
     * @throws FileNotFoundException 
     */
    public void trainFromExamples(int noOfIteration, double eenta) throws FileNotFoundException {
        //Use the wordcount to calcualte the weight verctor repetetively
        for (int count = 0; count < noOfIteration; count++) 
        {
            for (List<Integer> wordCount : wordCountSpamList) 
            {
                //Compute the new weight vector
                calculateNewWeightVector(eenta, true, wordCount);
                    
            }
            for (List<Integer> wordCount : wordCountHamList) 
            {
                //Compute the new weight vector
                calculateNewWeightVector(eenta, false, wordCount);
                
            }
        }
        
    }

    
    /**
     * Returns the accuracy over the test set
     * 
     * @return double array with accuracy in percentage 
     * @throws FileNotFoundException 
     */
    public double[] returnAccuracy() throws FileNotFoundException {
        File folder = new File(SPAMTEST);
        File[] listOfFiles = folder.listFiles();
        double percentage = 0;
        double[] returnValue = new double[2];

        //CALCULATE THE ACCURACY OF SPAM
        for (File spamFile : listOfFiles) {
            if (IsSpam(spamFile)) {
                percentage++;
            }
        }

        percentage = percentage * 100 / listOfFiles.length;
        returnValue[0] = percentage;

        //CALCULATE THE ACCURACY OF HAM
        percentage = 0;
        folder = new File(HAMTEST);
        listOfFiles = folder.listFiles();
        for (File hamFile : listOfFiles) {
            if (!IsSpam(hamFile)) {
                percentage++;
            }
        }

        percentage = percentage * 100 / listOfFiles.length;
        returnValue[1] = percentage;

        return returnValue;
    }

    /**
     * Tests if the test file is SPAM or HAM. If it is SPAM, returns True. Else return False.
     * 
     * @param testFile
     * @return
     * @throws FileNotFoundException 
     */
    public boolean IsSpam(File testFile) throws FileNotFoundException {
        List<String> words = readFile(testFile);
        List<Integer> wordCount = new ArrayList<>();
        int index;

        //Set the wordCount to zero for all words in vocabularyList.
        wordCount.add(1);
        for (int count = 0; count < vocabularyList.size(); count++) {
            wordCount.add(new Integer(0));
        }

        //Calculate wordCount for the testFile
        for (String eachWord : words) {
            if (this.vocabularyList.contains(eachWord)) {
                index = this.vocabularyList.indexOf(eachWord);
                wordCount.set(index, wordCount.get(index) + 1);
            }
        }

        //Calculate the value of perceptron function O
        return getVavlueOfO(wordCount) > 0;
    }

    /**
     * Calculates the new weight vector using the Perceptron weight update rule.
     *
     * @param eenta
     * @param isSpam
     * @param wordCount
     */
    public void calculateNewWeightVector(double eenta, boolean isSpam, List<Integer> wordCount) {

        double valueOfT;
        double valueOfO;
        double temp;
        if (isSpam) {
            valueOfT = 1.0;
        } else {
            valueOfT = -1.0;
        }

        valueOfO = getVavlueOfO(wordCount);

        if (valueOfO == valueOfT) {
          //return;
        } else {
            for (int count = 0; count < weights.size(); count++) {
                temp = weights.get(count) + eenta * (valueOfT - valueOfO) * wordCount.get(count);
                weights.set(count, temp);
            }
        }
    }

    /**
     *
     * Calculate the wordCount from the file
     *
     * @param file
     * @return wordCount for the file
     * @throws FileNotFoundException
     */
    public List<Integer> populateCountFromFile(File file) throws FileNotFoundException {
        int index;
        List<Integer> wordCount = new ArrayList<>();

        //Set the wordCount to zero for all words in vocabularyList.
        wordCount.add(1);
        for (int count = 0; count < vocabularyList.size(); count++) {
            wordCount.add(new Integer(0));
        }

        //Read all the words from the file
        List<String> words = readFile(file);

        //COMPUTE THE X vector for Perceptron weight displacement calculation
        for (String word : words) {

            //ADD THE WORD TO THE VOCABULARY LIST
            if (vocabularyList.contains(word)) {
                index = vocabularyList.indexOf(word);
                wordCount.set(index, wordCount.get(index) + 1);
            }
        }
        return wordCount;
    }

    /**
     * Calculates the value of Perceptron function
     *
     * @param wordCount
     * @return
     */
    public double getVavlueOfO(List<Integer> wordCount) {
        double result = 0.0;
        for (int i = 0; i < weights.size(); i++) {
            result = result + weights.get(i).doubleValue() * wordCount.get(i).doubleValue();
        }

        if (result > 0) {
            result = 1.0;
        } else {
            result = -1.0;
        }

        return result;
    }

    /**
     * Reads from the stop file and returns the list of all words in it
     *
     * @param stopFile
     * @return
     * @throws FileNotFoundException
     */
    private List<String> readStopFile(File stopFile) throws FileNotFoundException {

        Scanner scanner = new Scanner(stopFile);
        List<String> words;
        words = new ArrayList<>();
        String temp;

        while (scanner.hasNextLine()) {
            temp = scanner.nextLine();
            words.add(temp.toLowerCase());
        }

        return words;
    }

    /**
     * Read from the file specified and returns a list of all the individual
     * words in the document.
     *
     * @param spamFile
     * @return list of all valid words in the file
     * @throws FileNotFoundException
     */
    public List<String> readFile(File spamFile) throws FileNotFoundException {

        Scanner scanner = new Scanner(spamFile);
        List<String> words;
        words = new ArrayList<>();
        List<String> temp;
        List<String> tempStr;
        while (scanner.hasNextLine()) 
        {
            temp = Arrays.asList(scanner.nextLine().split("[ :@,.-]"));
            for (String t : temp) 
            {
                t = t.toLowerCase();
                if (!t.matches("[a-z']+") || (stopListEnable && stopWords.contains(t))) 
                {
                    continue;
                } 
                else 
                {
                    words.add(t);
                }
            }
        }

        return words;
    }

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException {

        //Perceptron without StopList
        Perceptron perceptronSLML;
        Perceptron perceptronML;
        try {
            double[] value;

            perceptronSLML = new Perceptron(args, true);
            
            List<Double> weights = perceptronSLML.trainPerceptronClassifier(12,0.8,null);

        } catch (FileNotFoundException ex) {
            Logger.getLogger(Perceptron.class.getName()).log(Level.SEVERE, null, ex);
            ex.printStackTrace();
        }
    }

}
