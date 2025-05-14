// Custom JavaScript for the Missing Persons Recognition System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  
    // Auto-dismissing alerts
    const alerts = document.querySelectorAll('.alert:not(.alert-important)');
    alerts.forEach(function(alert) {
      setTimeout(function() {
        const closeButton = alert.querySelector('.btn-close');
        if (closeButton) {
          closeButton.click();
        }
      }, 5000);
    });
  
    // File input validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(function(input) {
      input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
          // Check file type
          const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
          if (!validTypes.includes(file.type)) {
            alert('Invalid file type. Please upload a JPG, JPEG, or PNG image.');
            input.value = '';
            return;
          }
          
          // Check file size (max 5MB)
          const maxSize = 5 * 1024 * 1024; // 5MB
          if (file.size > maxSize) {
            alert('File size exceeds 5MB. Please upload a smaller image.');
            input.value = '';
            return;
          }
        }
      });
    });
  
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(function(form) {
      form.addEventListener('submit', function(event) {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        }
        form.classList.add('was-validated');
      }, false);
    });
  });
  
  // Function to show loading spinner
  function showLoading(buttonElement) {
    const originalText = buttonElement.innerHTML;
    buttonElement.disabled = true;
    buttonElement.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
    
    // Store original text for later restoration
    buttonElement.setAttribute('data-original-text', originalText);
    
    return originalText;
  }
  
  // Function to hide loading spinner and restore button
  function hideLoading(buttonElement) {
    const originalText = buttonElement.getAttribute('data-original-text');
    buttonElement.disabled = false;
    buttonElement.innerHTML = originalText;
  }
  
  // Add loading indicators to forms
  document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
      form.addEventListener('submit', function() {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
          showLoading(submitBtn);
        }
      });
    });
  });