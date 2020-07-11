package activeSegmentation.gui;

import activeSegmentation.ASCommon;
import activeSegmentation.IDeepLearning;
import activeSegmentation.deepLearning.UNetImplementation;
import activeSegmentation.learning.DeepLearningManager;
import activeSegmentation.prj.ProjectInfo;
import activeSegmentation.prj.ProjectManager;
import activeSegmentation.util.GuiUtil;
import weka.classifiers.AbstractClassifier;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class DeepLearningPanel implements Runnable, ASCommon {
    private JList<String> modelList;
    private GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();
    private String originalOptions;
    String originalClassifierName;
    private ProjectManager projectManager;
    private ProjectInfo projectInfo;
    final JFrame frame = new JFrame("LEARNING");
    // public static final Font FONT = new Font("Arial", 1, 13);
    JList<String> featureSelList;
    final ActionEvent COMPUTE_BUTTON_PRESSED = new ActionEvent(this, 1, "Compute");
    final ActionEvent SAVE_BUTTON_PRESSED = new ActionEvent(this, 2, "Save");
    DeepLearningManager deepLearningManager;

    public DeepLearningPanel(ProjectManager projectManager, DeepLearningManager deepLearningManager)  {
        this.projectManager = projectManager;
        this.deepLearningManager=deepLearningManager;
        this.projectInfo = projectManager.getMetaInfo();
        this.modelList = GuiUtil.model();
    }

    public void doAction(ActionEvent event)  {
        if (event == this.SAVE_BUTTON_PRESSED)     {
            //System.out.println(this.featureSelList.getSelectedIndex());
            this.projectInfo.setFeatureSelection((String)this.featureSelList.getSelectedValue());

            // System.out.println("in set classifiler");
            AbstractClassifier testClassifier=setClassifier();

            if(testClassifier!=null) {
                IDeepLearning deepModel = new UNetImplementation();
                this.deepLearningManager.setClassifier(deepModel);
                this.projectManager.updateMetaInfo(this.projectInfo);
            }

        }
    }

    public void run()  {
        this.frame.setDefaultCloseOperation(1);
        this.frame.getContentPane().setBackground(Color.GRAY);
        this.frame.setLocationRelativeTo(null);
        this.frame.setSize(600, 250);
        JPanel learningP = new JPanel();
        learningP.setLayout(null);
        learningP.setBackground(Color.GRAY);

        JPanel learningJPanel = new JPanel();
        learningJPanel.setBorder(BorderFactory.createTitledBorder("Learning"));

        PropertyPanel m_CEPanel = new PropertyPanel(this.m_ClassifierEditor);
        this.m_ClassifierEditor.setClassType(UNetImplementation.class);
        this.m_ClassifierEditor.setValue(this.deepLearningManager.getClassifier());
        Object c = this.m_ClassifierEditor.getValue();
        originalOptions = "";
        this.originalClassifierName = c.getClass().getName();
        if ((c instanceof OptionHandler)) {
            originalOptions = Utils.joinOptions(((OptionHandler)c).getOptions());
        }
        m_CEPanel.setBounds(30, 30, 250, 30);
        learningJPanel.add(m_CEPanel);
        learningJPanel.setBounds(10, 20, 300, 80);

        JPanel featureSelection = new JPanel();
        featureSelection.setBorder(BorderFactory.createTitledBorder("Feature Selection"));
        featureSelection.setBounds(370, 20, 200, 80);
        DefaultListModel<String> model = new DefaultListModel<String>();
        this.featureSelList = new JList<String>(model);
        model.addElement("NONE");
        model.addElement("Principle Component Analysis");
        model.addElement("Correlation Based Selection");
        this.featureSelList.setBackground(Color.WHITE);
        this.featureSelList.setSelectedIndex(0);
        featureSelection.add(this.featureSelList);

        JPanel options = new JPanel();
        options.setBorder(BorderFactory.createTitledBorder("Learning Options"));
        CheckboxGroup checkboxGroup = new CheckboxGroup();
        options.setBounds(10, 120, 300, 80);

        Checkbox pasiveLearning = new Checkbox("Passive Learning", checkboxGroup, true);
        Checkbox activeLearning = new Checkbox("Active Learning", checkboxGroup, true);
        options.add(pasiveLearning);
        options.add(activeLearning);
        JPanel resetJPanel = new JPanel();
        resetJPanel.setBackground(Color.GRAY);
        resetJPanel.setBounds(370, 120, 200, 80);
        resetJPanel.add(addButton("SAVE", null, 370, 120, 200, 50, this.SAVE_BUTTON_PRESSED));

        learningP.add(learningJPanel);
        learningP.add(featureSelection);
        learningP.add(resetJPanel);
        learningP.add(options);

        this.frame.add(learningP);
        this.frame.setVisible(true);
    }

    private AbstractClassifier setClassifier()
    {
        Object c = this.m_ClassifierEditor.getValue();
        String options = "";
        String[] optionsArray = ((OptionHandler)c).getOptions();
        System.out.println(originalOptions);
        if ((c instanceof OptionHandler)) {
            options = Utils.joinOptions(optionsArray);
        }
        if ((!this.originalClassifierName.equals(c.getClass().getName())) ||
                (!this.originalOptions.equals(options))) {
            try
            {
                AbstractClassifier cls = (AbstractClassifier)c.getClass().newInstance();
                cls.setOptions(optionsArray);
                return cls;
            }
            catch (Exception ex)
            {
                ex.printStackTrace();
            }
        }
        return null;
    }

    private JButton addButton(String label, ImageIcon icon, int x, int y, int width, int height, final ActionEvent action)
    {
        JButton button = new JButton(label, icon);
        button.setFont(labelFONT);
        button.setBorderPainted(false);
        button.setFocusPainted(false);
        button.setBackground(new Color(192, 192, 192));
        button.setForeground(Color.WHITE);
        button.setBounds(x, y, width, height);
        button.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                DeepLearningPanel.this.doAction(action);
            }
        });
        return button;
    }
}