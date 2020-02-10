from django.conf import settings
from django.shortcuts import render
from .forms import QueryForm
import os
from diabetic_retinopathy.test import Test
import shutil


def get_test_case_path():
    return ['TestCase/{}.jpg'.format(str(i)) for i in range(1, 41)]


def index_view(request):
    if request.method == 'POST':
        query = QueryForm(request.POST, request.FILES)
        if query.is_valid():
            handle_uploaded_file(request.FILES['query'])
            return render(request, 'index.html', {'query_url': str(request.FILES['query']),
                                                  'test_cases': get_test_case_path()})
        else:
            return render(request, "index.html", {'test_cases': get_test_case_path()})
    elif request.method == 'GET':
        if 'query_url' in request.GET:
            query_name = str.strip(request.GET['query_url'])
            query_name320 = query_name[:-3]+'320.jpg'
            classifier = request.GET['classifier']
            test = Test(classifier)
            test.test(query_name)
            return render(request, "index.html", {'results': True, 'query_url': query_name,
                                                  'query_url320': query_name320,
                                                  'test_cases': get_test_case_path()})
        elif 'test_sample' in request.GET:
            test_path = settings.BASE_DIR+'/static/'+request.GET['test_sample']
            test_name = request.GET['test_sample'].split('/')[-1]
            handle_uploaded_file_test(test_path, test_name)
            return render(request, 'index.html', {'query_url': test_name,
                                                  'test_cases': get_test_case_path()})
        else:
            return render(request, 'index.html', {'test_cases': get_test_case_path()})
    else:
        return render(request, "index.html", {'test_cases': get_test_case_path()})


def handle_uploaded_file(f):
    if not os.path.isdir(settings.MEDIA_ROOT+'/query'):
        os.mkdir(settings.MEDIA_ROOT+'/query')

    with open(settings.MEDIA_ROOT+'/query/'+f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def handle_uploaded_file_test(f, f_n):
    if not os.path.isdir(settings.MEDIA_ROOT+'/query'):
        os.mkdir(settings.MEDIA_ROOT+'/query')

    shutil.copy(f, settings.MEDIA_ROOT+'/query/'+f_n)
