from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER

# Output PDF path
output_path = "NHANES_Model_Transitions.pdf"

# Create canvas
c = canvas.Canvas(output_path, pagesize=LETTER)
width, height = LETTER
y = height - 50  # starting height

# Title
c.setFont("Helvetica-Bold", 14)
c.drawCentredString(width / 2, y, "Model Transition Explanation – NHANES BMI Prediction")
y -= 30

# Set up transition text content
transitions = [
    ("OLS -> Ridge Regression", 
     "OLS gave us a strong starting point but suffered from multicollinearity issues due to correlated features. Ridge Regression added L2 regularization, which controlled the size of coefficients and improved model robustness."),

    ("Ridge -> Lasso Regression", 
     "While Ridge controlled overfitting, it didn’t simplify the model. Lasso Regression added L1 penalty which performed feature selection by setting some coefficients to zero, leading to a more interpretable model with slightly improved test performance."),

    ("Lasso -> Regression Tree", 
     "Linear models assume linear relationships. To capture non-linear patterns in BMI prediction, we moved to Regression Trees, which split the data into regions using thresholds. Although they captured complexity, they were prone to overfitting."),

    ("Regression Tree -> Random Forest", 
     "Random Forest overcame overfitting by averaging multiple trees trained on different data samples. This ensemble approach improved generalization and gave smoother predictions, though at the cost of interpretability."),

    ("Random Forest -> XGBoost", 
     "XGBoost improved upon Random Forest by building trees sequentially, focusing on errors from previous trees. It handled missing data, offered regularization, and gave one of the highest accuracies, though hyperparameter tuning was complex."),

    ("XGBoost -> Bayesian Linear Regression", 
     "While XGBoost performed well, it lacked uncertainty estimation. Bayesian Linear Regression offered confidence intervals with each prediction, making it suitable for medical applications. Though performance dropped slightly, it added interpretability and trust."),

    ("All Models -> Linear Regression + Interaction Terms", 
     "Finally, we introduced interaction terms like Waist × Weight in our linear models. This feature engineering step dramatically improved performance, even surpassing complex models like XGBoost. It proved that well-engineered features can beat high-complexity models.")
]

# Set font for content
c.setFont("Helvetica", 11)
line_height = 16

# Write content
for title, body in transitions:
    if y < 100:
        c.showPage()
        y = height - 50
        c.setFont("Helvetica", 11)

    # Title
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, title)
    y -= line_height

    # Body
    c.setFont("Helvetica", 11)
    for line in body.split(". "):
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
        c.drawString(60, y, line.strip() + ("" if line.strip().endswith(".") else "."))
        y -= line_height

# Save PDF
c.save()
print(f"PDF generated successfully at: {output_path}")
